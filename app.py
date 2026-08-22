"""
ContextMesh — End-to-End Multimodal RAG Pipeline

Question → load_evidence → search_evidence → expand_relationships → Grounded Context → Groq → GroundedAnswer

Usage:
    python app.py                              # Run the golden demo query
    python app.py "Your question here"         # Run a custom query
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

from knowledge.schema import Evidence, GroundedAnswer
from knowledge.relationships import expand_relationships
from retrieval.search import search_evidence
from retrieval.grounding import build_grounded_prompt, build_grounded_answer, format_provenance
from llm.groq_client import query_groq
from tests.test_end_to_end import load_evidence


EVIDENCE_PATH = os.path.join(root_dir, "data", "processed", "evidence.json")

GOLDEN_QUERY = (
    "What architecture was proposed to reduce database load, "
    "and what visual/document evidence supports it?"
)


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def ask(
    question: str,
    top_k: int = 2,
    max_hops: int = 1,
    verbose: bool = False,
) -> GroundedAnswer:
    """
    Full ContextMesh pipeline matching Person 2's API contract:

        app.py
           ↓
        load evidence
           ↓
        search_evidence(query, all_evidence, top_k=2)
           ↓
        seed Evidence objects
           ↓
        expand_relationships(seed, all_evidence, max_hops=1)
           ↓
        build grounded context
           ↓
        Groq
           ↓
        GroundedAnswer
    """
    all_evidence = load_evidence()
    if not all_evidence:
        return GroundedAnswer(
            query=question,
            answer="No evidence found in data/processed/evidence.json. Please run pipeline/ingest.py first.",
            cited_evidence=[],
            modalities_used=[],
            metadata={"status": "error_no_evidence"}
        )

    # 1. Direct Search Hits
    if verbose:
        print("\n[1/4] Running search_evidence (direct retrieval)...")

    seeds = search_evidence(question, all_evidence, top_k=top_k)

    if verbose:
        print(f"      Top direct hits (top_k={top_k}): {[ev.id for ev in seeds]}")
        for ev in seeds:
            print(f"        [{ev.modality:12s}] {ev.id} ({ev.source})")

    # 2. Relationship Expansion
    if verbose:
        print("\n[2/4] Running expand_relationships (graph traversal)...")

    expanded_evidence = expand_relationships(seeds, all_evidence, max_hops=max_hops)

    if verbose:
        print(f"      Expanded context ({len(expanded_evidence)} items): {[ev.id for ev in expanded_evidence]}")

    # 3. Build Grounded Context Prompt
    if verbose:
        print("\n[3/4] Building grounded context prompt...")

    system_prompt, user_prompt = build_grounded_prompt(question, expanded_evidence)

    # 4. Groq LLM Call
    if verbose:
        print("\n[4/4] Calling Groq LLM...")

    try:
        raw_answer = query_groq(system_prompt, user_prompt)
    except EnvironmentError as e:
        raw_answer = f"[Groq API Key missing] {e}"
    except RuntimeError as e:
        raw_answer = f"[Groq API Error] {e}"

    # 5. Build & Return GroundedAnswer Contract Object
    grounded_answer = build_grounded_answer(
        query=question,
        answer_text=raw_answer,
        cited_evidence=expanded_evidence,
        metadata={
            "seed_ids": [ev.id for ev in seeds],
            "expanded_ids": [ev.id for ev in expanded_evidence],
            "top_k": top_k,
            "max_hops": max_hops,
        }
    )

    return grounded_answer


# ---------------------------------------------------------------------------
# Pretty Printer
# ---------------------------------------------------------------------------

def print_result(result: GroundedAnswer) -> None:
    """Print the GroundedAnswer in a clean human-readable format."""
    print("\n" + "=" * 70)
    print("  ContextMesh -- Multimodal RAG Grounded Answer")
    print("=" * 70)

    print(f"\nQuestion: {result.query}")
    print(f"\n{'-' * 70}")
    print(f"\n{result.answer}")
    print(f"\n{'-' * 70}")

    if result.cited_evidence:
        print(f"\nCited Evidence ({len(result.cited_evidence)} items, Modalities: {', '.join(result.modalities_used)}):")
        for ev in result.cited_evidence:
            line = f"  * {ev.source}"
            if ev.timestamp is not None:
                mins, secs = divmod(int(ev.timestamp), 60)
                line += f" -- {mins:02d}:{secs:02d}"
            if ev.page is not None:
                line += f" -- page {ev.page}"
            line += f"  [{ev.modality}] ({ev.id})"
            print(line)
    else:
        print("\n  (no cited evidence)")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = GOLDEN_QUERY

    result = ask(question, verbose=True)
    print_result(result)
