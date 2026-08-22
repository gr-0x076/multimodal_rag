"""
ContextMesh — Grounding Module

Constructs the grounded LLM prompt from retrieved Evidence objects.
Ensures the model answers ONLY from supplied evidence and preserves
provenance (source, timestamp, page).
"""

import sys
from pathlib import Path
from typing import List, Tuple

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence, GroundedAnswer


# ---------------------------------------------------------------------------
# GroundedAnswer Constructor — Person 2 Contract
# ---------------------------------------------------------------------------

def build_grounded_answer(
    query: str,
    answer_text: str,
    cited_evidence: List[Evidence],
    engine: str = "fallback",
    metadata: dict = None,
) -> GroundedAnswer:
    """
    Construct a GroundedAnswer object conforming to Person 2's evaluation contract.
    """
    modalities = sorted(list({ev.modality for ev in cited_evidence}))
    merged_metadata = {"engine": engine}
    if metadata:
        merged_metadata.update(metadata)

    return GroundedAnswer(
        query=query,
        answer=answer_text,
        cited_evidence=cited_evidence,
        modalities_used=modalities,
        metadata=merged_metadata,
    )


# ---------------------------------------------------------------------------
# System Prompt — grounding instructions for the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are ContextMesh, a multimodal evidence-grounded assistant.

RULES — you MUST follow every one of these:
1. Answer ONLY using the evidence provided below. Do NOT use prior knowledge.
2. Cite every claim with the evidence source, including:
   - The source filename
   - The timestamp (MM:SS) for audio/video evidence
   - The page number for PDF evidence
   - The evidence ID
3. If the supplied evidence is insufficient to answer the question,
   say: "The available evidence is insufficient to fully answer this question."
   Do NOT invent or fabricate information.
4. NEVER invent timestamps, page numbers, or sources that are not in the evidence.
5. Structure your answer clearly with a summary followed by an "Evidence" section
   listing each piece of supporting evidence.

Response format:
[Your grounded answer here, citing evidence inline.]

Evidence:
• [source] — [timestamp/page] — [brief description]
• [source] — [timestamp/page] — [brief description]
..."""


# ---------------------------------------------------------------------------
# Context Formatting
# ---------------------------------------------------------------------------

def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def format_evidence_context(evidence_list: List[Evidence]) -> str:
    """
    Format a list of Evidence objects into a structured text block
    that the LLM can reason over.

    Each evidence entry includes:
      - Evidence ID
      - Modality
      - Source file
      - Timestamp (if applicable)
      - Page (if applicable)
      - Content
      - Related evidence IDs
    """
    if not evidence_list:
        return "[No evidence available]"

    blocks = []
    for i, ev in enumerate(evidence_list, 1):
        lines = [
            f"--- Evidence {i} ---",
            f"ID       : {ev.id}",
            f"Modality : {ev.modality}",
            f"Source   : {ev.source}",
        ]

        if ev.timestamp is not None:
            lines.append(f"Timestamp: {_format_timestamp(ev.timestamp)} ({ev.timestamp}s)")
        if ev.page is not None:
            lines.append(f"Page     : {ev.page}")
        if ev.entities:
            lines.append(f"Entities : {', '.join(ev.entities)}")

        lines.append(f"Content  : {ev.content}")

        if ev.relationships:
            lines.append(f"Related  : {', '.join(ev.relationships)}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_grounded_prompt(
    question: str,
    evidence_list: List[Evidence],
) -> Tuple[str, str]:
    """
    Build the (system_message, user_message) pair for the LLM.

    Args:
        question:       The user's natural-language question.
        evidence_list:  Retrieved & relationship-expanded Evidence objects.

    Returns:
        (system_prompt, user_prompt)  ready to send to Groq.
    """
    context = format_evidence_context(evidence_list)

    user_prompt = (
        f"EVIDENCE:\n"
        f"{context}\n\n"
        f"QUESTION:\n"
        f"{question}\n\n"
        f"Answer the question using ONLY the evidence above. "
        f"Cite sources with their timestamps or page numbers."
    )

    return SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Provenance Formatter
# ---------------------------------------------------------------------------

def format_provenance(evidence_list: List[Evidence]) -> List[dict]:
    """
    Extract structured provenance metadata from Evidence objects.
    Used to build the sources section of the final response.
    """
    sources = []
    for ev in evidence_list:
        entry = {
            "id": ev.id,
            "source": ev.source,
            "modality": ev.modality,
        }
        if ev.timestamp is not None:
            entry["timestamp"] = _format_timestamp(ev.timestamp)
            entry["timestamp_seconds"] = ev.timestamp
        if ev.page is not None:
            entry["page"] = ev.page
        sources.append(entry)
    return sources


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_evidence = [
        Evidence(
            id="meeting_audio_2", modality="audio", source="meeting.mp4",
            content="We propose using Redis caching to reduce database load.",
            timestamp=10.0, entities=["Redis", "caching", "database"],
            relationships=["meeting_frame_10", "architecture_page_2"],
        ),
        Evidence(
            id="meeting_frame_10", modality="video_frame", source="meeting.mp4",
            content="Redis architecture diagram showing Application, Redis Cache, Database flow.",
            timestamp=10.0, entities=["Redis", "architecture"],
            relationships=["meeting_audio_2"],
        ),
        Evidence(
            id="architecture_page_2", modality="pdf", source="architecture.pdf",
            content="Redis is deployed as an in-memory caching layer between the application and database.",
            page=2, entities=["Redis", "caching", "database"],
            relationships=["meeting_audio_2"],
        ),
    ]

    system_msg, user_msg = build_grounded_prompt(
        "What architecture was proposed to reduce database load?",
        demo_evidence,
    )

    print("=== SYSTEM PROMPT ===")
    print(system_msg)
    print("\n=== USER PROMPT ===")
    print(user_msg)
    print("\n=== PROVENANCE ===")
    for src in format_provenance(demo_evidence):
        print(f"  {src}")
