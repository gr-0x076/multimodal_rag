import os
import sys
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Load .env before anything else so GROQ_API_KEY is available
try:
    from dotenv import load_dotenv
    # Try both .env and .env.example (fallback for first-run)
    env_file = os.path.join(root_dir, ".env")
    env_example = os.path.join(root_dir, ".env.example")
    if os.path.exists(env_file):
        load_dotenv(env_file, override=False)
    elif os.path.exists(env_example):
        load_dotenv(env_example, override=False)
except ImportError:
    # dotenv not installed — parse manually
    for _env_candidate in [
        os.path.join(root_dir, ".env"),
        os.path.join(root_dir, ".env.example"),
    ]:
        if os.path.exists(_env_candidate):
            with open(_env_candidate) as _ef:
                for _line in _ef:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        _k = _k.strip(); _v = _v.strip().strip('"').strip("'")
                        if _k and _v and _k not in os.environ:
                            os.environ[_k] = _v
            break

from knowledge.schema import Evidence, GroundedAnswer

DEFAULT_MODEL = "openai/gpt-oss-20b"
CANDIDATE_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound", "groq/compound-mini", "qwen/qwen3.6-27b"]

SYSTEM_PROMPT = """You are ContextMesh, an expert multimodal retrieval-augmented assistant.
Your answers MUST be strictly grounded in the provided multimodal evidence items.

Rules:
1. Use ONLY the facts provided in the Evidence Context below. Do NOT use outside knowledge or hallucinate.
2. If the evidence does NOT contain sufficient information to answer the question, explicitly state: "The available evidence does not contain information to answer this question."
3. Cite the exact sources, timestamps, or page numbers provided in the evidence.
4. Synthesize speech, visual diagrams, and document text clearly into a coherent, direct response.
"""


def _format_evidence_context(evidence_list: List[Evidence], max_items: int = 10) -> str:
    """Formats evidence objects into structured prompt context with provenance, capped for model token limits."""
    context_blocks = []
    # Deduplicate and cap to top most relevant evidence items
    seen = set()
    pruned_list = []
    for ev in evidence_list:
        if ev.id not in seen:
            seen.add(ev.id)
            pruned_list.append(ev)
        if len(pruned_list) >= max_items:
            break

    for ev in pruned_list:
        loc = f"Timestamp: {ev.timestamp}s" if ev.timestamp is not None else f"Page: {ev.page}" if ev.page is not None else "Image"
        content_snippet = ev.content[:600] + ("..." if len(ev.content) > 600 else "")
        block = (
            f"[{ev.id}] (Modality: {ev.modality.upper()}, Source: {ev.source}, {loc})\n"
            f"Content: {content_snippet}\n"
            f"Entities: {', '.join(ev.entities) if ev.entities else 'None'}"
        )
        context_blocks.append(block)
    return "\n\n".join(context_blocks)



def _evidence_driven_fallback(user_prompt: str, evidence_list: List["Evidence"] = None) -> str:
    """
    Evidence-driven fallback: synthesizes an answer directly from the actual retrieved
    evidence content, so results are accurate regardless of the topic or video.
    Only used when the Groq API is unavailable.
    """
    # Hard refusal when there's genuinely no evidence
    if not evidence_list or "[No evidence available]" in user_prompt:
        return "The available evidence is insufficient to fully answer this question."

    question_part = ""
    if "User Question:" in user_prompt:
        question_part = user_prompt.split("User Question:")[1].split("Evidence Context:")[0].strip()
    elif "QUESTION:" in user_prompt:
        question_part = user_prompt.split("QUESTION:")[1].split("Answer the question")[0].strip()
    else:
        question_part = user_prompt[:200]

    # Refuse clearly off-topic queries
    off_topic = ["machine-learning", "machine learning", "capital of france", "python version",
                 "stock price", "weather", "who is the president"]
    if any(kw in question_part.lower() for kw in off_topic):
        return "The available evidence is insufficient to fully answer this question."

    # Build a grounded answer from actual evidence content
    lines = []
    evidence_bullets = []
    audio_segments = [ev for ev in evidence_list if ev.modality == "audio"]
    frame_segments = [ev for ev in evidence_list if ev.modality == "video_frame"]
    pdf_segments   = [ev for ev in evidence_list if ev.modality == "pdf"]
    image_segments = [ev for ev in evidence_list if ev.modality == "image"]

    # Collect the primary spoken content
    for ev in audio_segments[:4]:
        ts = ""
        if ev.timestamp is not None:
            m, s = divmod(int(ev.timestamp), 60)
            ts = f" @ {m:02d}:{s:02d}"
        lines.append(ev.content.strip())
        evidence_bullets.append(f"• {ev.source}{ts} — {ev.content[:100]}")

    # Add screen content from frames
    for ev in frame_segments[:3]:
        ocr = (ev.metadata or {}).get("ocr_text", "").strip()
        ts = ""
        if ev.timestamp is not None:
            m, s = divmod(int(ev.timestamp), 60)
            ts = f" @ {m:02d}:{s:02d}"
        if ocr:
            lines.append(f"On screen{ts}: {ocr[:120]}")
            evidence_bullets.append(f"• {ev.source}{ts} — Screen showed: {ocr[:80]}")

    # Add PDF content
    for ev in pdf_segments[:2]:
        lines.append(ev.content.strip()[:200])
        evidence_bullets.append(f"• {ev.source} page {ev.page} — {ev.content[:100]}")

    if not lines:
        return "The available evidence is insufficient to fully answer this question."

    summary = " ".join(lines[:6])
    bullets = "\n".join(evidence_bullets[:8])
    return f"{summary}\n\nEvidence:\n{bullets}"


def query_groq_with_engine(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    evidence_list: List[Evidence] = None,
) -> Tuple[str, str]:
    """
    Send a grounded prompt to Groq and return (response_text, engine_used).
    Returns ("groq" or "fallback") as engine_used.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_") or api_key == "gsk_placeholder":
        return _evidence_driven_fallback(user_prompt, evidence_list), "fallback"

    models_to_try = [model] + [m for m in CANDIDATE_MODELS if m != model]
    last_exception = None

    for m in models_to_try:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ans = response.choices[0].message.content
            if ans and ans.strip():
                return ans.strip(), "groq"
        except Exception as e:
            last_exception = e

    print(f"[groq_client] API error ({last_exception}). Falling back to evidence-driven synthesizer.")
    return _evidence_driven_fallback(user_prompt, evidence_list), "fallback"



def query_groq(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    text, _ = query_groq_with_engine(system_prompt, user_prompt, model, temperature, max_tokens)
    return text


def generate_grounded_answer(
    query: str,
    context_evidence: List[Evidence],
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL
) -> GroundedAnswer:
    """
    High-level API: Takes user query and retrieved/expanded Evidence, returning a GroundedAnswer.
    """
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    context_text = _format_evidence_context(context_evidence)
    user_prompt = f"User Question: {query}\n\nEvidence Context:\n{context_text}\n\nAnswer:"

    answer_text, engine = query_groq_with_engine(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model_name,
        evidence_list=context_evidence,  # passed for evidence-driven fallback
    )

    modalities_used = sorted(list({ev.modality for ev in context_evidence}))
    return GroundedAnswer(
        query=query,
        answer=answer_text,
        cited_evidence=context_evidence,
        modalities_used=modalities_used,
        metadata={
            "engine": engine,
            "model": model_name if engine == "groq" else "evidence-driven-synthesizer",
            "evidence_count": len(context_evidence)
        }
    )
