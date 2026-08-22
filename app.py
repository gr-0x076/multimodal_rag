"""
ContextMesh — End-to-End Multimodal RAG Pipeline

Question → Retrieval → Relationship Expansion → Grounding → Groq → Answer + Provenance

Usage:
    python app.py                              # Run the golden demo query
    python app.py "Your question here"         # Run a custom query
    streamlit run app.py                       # Launch the Streamlit UI (later)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from knowledge.relationships import build_relationship_graph, expand_related
from retrieval.search import search, load_evidence
from retrieval.grounding import build_grounded_prompt, format_provenance
from llm.groq_client import query_groq


EVIDENCE_PATH = os.path.join(root_dir, "data", "processed", "evidence.json")

GOLDEN_QUERY = (
    "What architecture was proposed to reduce database load, "
    "and where is the supporting visual/document evidence?"
)


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def ask(
    question: str,
    evidence_path: Optional[str] = None,
    top_k: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Full ContextMesh pipeline:

        Question
            ↓
        Retrieval  (search across audio, video_frame, pdf, image)
            ↓
        Relationship expansion
            ↓
        Grounded context construction
            ↓
        Groq LLM call
            ↓
        Answer + Provenance

    Returns:
        {
            "question": str,
            "answer": str,
            "sources": [
                {"id": ..., "source": ..., "modality": ..., "timestamp": ..., "page": ...},
                ...
            ],
            "evidence_count": int,
        }
    """
    evidence_path = evidence_path or EVIDENCE_PATH

    # Step 1 — Retrieval: find relevant evidence across all modalities
    if verbose:
        print("\n[1/4] Retrieving relevant evidence...")

    retrieved = search(question, evidence_path=evidence_path, top_k=top_k)

    if not retrieved:
        return {
            "question": question,
            "answer": "No evidence found. Please ensure the evidence store has been populated.",
            "sources": [],
            "evidence_count": 0,
        }

    if verbose:
        print(f"      Found {len(retrieved)} evidence items:")
        for ev in retrieved:
            label = f"{ev.source}"
            if ev.timestamp is not None:
                mins, secs = divmod(int(ev.timestamp), 60)
                label += f" @ {mins:02d}:{secs:02d}"
            if ev.page is not None:
                label += f" — page {ev.page}"
            print(f"        [{ev.modality:12s}] {ev.id}  ({label})")

    # Step 2 — Build grounded prompt
    if verbose:
        print("\n[2/4] Building grounded context...")

    system_prompt, user_prompt = build_grounded_prompt(question, retrieved)

    if verbose:
        print(f"      Context length: {len(user_prompt)} chars")

    # Step 3 — Call Groq
    if verbose:
        print("\n[3/4] Calling Groq LLM...")

    try:
        answer = query_groq(system_prompt, user_prompt)
    except EnvironmentError as e:
        return {
            "question": question,
            "answer": f"[Groq API not configured] {e}",
            "sources": format_provenance(retrieved),
            "evidence_count": len(retrieved),
        }
    except RuntimeError as e:
        return {
            "question": question,
            "answer": f"[Groq API error] {e}",
            "sources": format_provenance(retrieved),
            "evidence_count": len(retrieved),
        }

    # Step 4 — Format provenance
    if verbose:
        print("\n[4/4] Formatting provenance...")

    sources = format_provenance(retrieved)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "evidence_count": len(retrieved),
    }


# ---------------------------------------------------------------------------
# Pretty Printer
# ---------------------------------------------------------------------------

def print_result(result: Dict[str, Any]) -> None:
    """Print the pipeline result in a human-readable format."""
    print("\n" + "=" * 70)
    print("  ContextMesh -- Multimodal RAG Response")
    print("=" * 70)

    print(f"\nQuestion: {result['question']}")
    print(f"\n{'-' * 70}")
    print(f"\n{result['answer']}")
    print(f"\n{'-' * 70}")

    if result["sources"]:
        print(f"\nEvidence Sources ({result['evidence_count']} items):")
        for src in result["sources"]:
            line = f"  * {src['source']}"
            if "timestamp" in src:
                line += f" -- {src['timestamp']}"
            if "page" in src:
                line += f" -- page {src['page']}"
            line += f"  [{src['modality']}]"
            print(line)
    else:
        print("\n  (no evidence found)")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Accept a custom query or use the golden demo query
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = GOLDEN_QUERY

    # Check if evidence exists; if not, try generating seed data
    if not os.path.exists(EVIDENCE_PATH):
        print("[setup] evidence.json not found. Generating seed data...")
        try:
            from tests.create_seed_data import main as create_seed
            create_seed()
        except Exception as e:
            print(f"[setup] Could not generate seed data: {e}")
            print(f"[setup] Run: python -m tests.create_seed_data")
            sys.exit(1)

    result = ask(question, verbose=True)
    print_result(result)
