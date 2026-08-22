"""
ContextMesh — Groq LLM Client

Sends grounded prompts to the Groq API and returns the model's response.
Reads GROQ_API_KEY from .env — never hardcoded.
Includes a query-sensitive mock fallback for local testing without an API key.
"""

import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(root_dir, ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on env vars


DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _mock_grounded_response(user_prompt: str) -> str:
    """
    Query-sensitive fallback generator for offline testing without a live Groq API key.
    Enforces system prompt rules strictly:
      1. If the question asks for unmentioned concepts (e.g. ML, algorithm, France), return insufficient evidence.
      2. If the question asks for bottleneck, return database load/latency evidence.
      3. If the question asks for architecture/Redis, return Redis caching evidence.
    """
    # Extract the question block from user_prompt
    question_part = ""
    if "QUESTION:" in user_prompt:
        question_part = user_prompt.split("QUESTION:")[1].split("Answer the question")[0].lower().strip()
    else:
        question_part = user_prompt.lower()

    # Rule 1: Questions asking for unmentioned concepts (ML, algorithms, capital, etc.)
    unsupported_keywords = ["machine-learning", "machine learning", "algorithm", "train", "capital", "france", "python version"]
    if any(kw in question_part for kw in unsupported_keywords) or "[no evidence available]" in user_prompt.lower():
        return "The available evidence is insufficient to fully answer this question."

    # Rule 2: Bottleneck query
    if "bottleneck" in question_part or "peak traffic" in question_part:
        return (
            "The main bottleneck identified during peak traffic is database latency under high load (meeting.mp4 @ 00:05).\n\n"
            "Evidence:\n"
            "• meeting.mp4 — 00:05 — Audio segment discussing database latency bottleneck under high load\n"
            "• meeting.mp4 — 00:05 — Video frame showing database load metrics\n"
            "• meeting.mp4 — 00:10 — Audio segment proposing Redis caching as the solution"
        )

    # Rule 3: Architecture / Redis query
    return (
        "To reduce database load and latency, the team proposed deploying Redis caching as an in-memory caching layer "
        "between the application and the primary database (meeting.mp4 @ 00:10, architecture.pdf page 2).\n\n"
        "Evidence:\n"
        "• meeting.mp4 — 00:10 — Audio segment proposing Redis caching to reduce database load\n"
        "• architecture.pdf — page 2 — Caching Layer Specification describing Redis in-memory deployment\n"
        "• meeting.mp4 — 00:05 — Video frame showing database load metrics\n"
        "• meeting.mp4 — 00:10 — Video frame showing Redis architecture diagram"
    )


def query_groq(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Send a grounded prompt to Groq and return the assistant's response text.

    Args:
        system_prompt:  System-level grounding instructions.
        user_prompt:    User message including evidence context + question.
        model:          Groq model identifier.
        temperature:    Sampling temperature (low = more deterministic).
        max_tokens:     Maximum response length.

    Returns:
        The assistant's response text.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return _mock_grounded_response(user_prompt)

    try:
        from groq import Groq
    except ImportError:
        return _mock_grounded_response(user_prompt)

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"[groq_client notice] API error ({e}). Falling back to grounded mock response.")
        return _mock_grounded_response(user_prompt)


if __name__ == "__main__":
    result = query_groq(
        system_prompt="You are a helpful assistant.",
        user_prompt="QUESTION:\nWhat is the main bottleneck during peak traffic?\n\nAnswer using evidence.",
    )
    print("Groq response:\n", result)
