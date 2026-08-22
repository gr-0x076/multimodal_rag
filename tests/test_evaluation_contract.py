import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence, GroundedAnswer
from knowledge.relationships import expand_relationships
from retrieval.search import search_evidence
from tests.test_end_to_end import load_evidence


def test_retrieval_and_relationship_expansion_contract():
    """
    Validates Person 3's API Contracts:
      1. search_evidence(query, all_evidence, top_k) -> List[Evidence]
      2. expand_relationships(seed_evidence, all_evidence, max_hops) -> List[Evidence]
      3. GroundedAnswer contract format
    """
    print("\n--- Evaluation Test 1: Person 3 Retrieval & Expansion Contract ---")
    all_evidence = load_evidence()
    query = "What architecture was proposed to reduce database load?"
    
    # 1. Direct Search (Top 2 Seeds)
    seeds = search_evidence(query, all_evidence, top_k=2)
    seed_ids = [ev.id for ev in seeds]
    print(f"Top direct hits (top_k=2): {seed_ids}")
    
    assert len(seeds) > 0, "search_evidence must return relevant seeds"
    assert "meeting_audio_2" in seed_ids or "architecture_page_2" in seed_ids, \
        "Expected top direct hits to include core spoken or written proposal"
        
    # 2. Relationship Expansion (1-hop traversal)
    grounded_context = expand_relationships(seeds, all_evidence, max_hops=1)
    context_ids = [ev.id for ev in grounded_context]
    context_modalities = {ev.modality for ev in grounded_context}
    
    print(f"Expanded multimodal context ({len(grounded_context)} items): {context_ids}")
    print(f"Context modalities covered: {context_modalities}")
    
    # Assert cross-modal coverage in the grounded context
    assert "audio" in context_modalities, "Grounded context must contain audio evidence"
    assert "video_frame" in context_modalities, "Grounded context must contain visual video frame evidence"
    assert "pdf" in context_modalities, "Grounded context must contain PDF specification evidence"
    
    # 3. Grounded Answer Construction Contract
    mock_answer = GroundedAnswer(
        query=query,
        answer="The team proposed Redis caching in front of the database to reduce load and latency.",
        cited_evidence=grounded_context,
        modalities_used=sorted(list(context_modalities)),
        metadata={"seed_count": len(seeds), "expanded_count": len(grounded_context)}
    )
    answer_dict = mock_answer.to_dict()
    assert answer_dict["query"] == query
    assert len(answer_dict["cited_evidence"]) == len(grounded_context)
    assert len(answer_dict["modalities_used"]) >= 3
    print(" Person 3 Retrieval, Expansion, and GroundedAnswer contracts validated successfully!")


def test_baseline_vs_multimodal_rag_comparison():
    """
    Evaluates Text-Only Baseline RAG vs. ContextMesh Multimodal Graph RAG.
    Validates that ContextMesh recovers visual frames & documents missed by text-only RAG.
    """
    print("\n--- Evaluation Test 2: Baseline vs Multimodal RAG Comparison ---")
    all_evidence = load_evidence()
    query = "What architecture was proposed to reduce database load?"
    
    # Baseline: Transcript text only
    transcript_only = [ev for ev in all_evidence if ev.modality == "audio"]
    baseline_hits = search_evidence(query, transcript_only, top_k=2)
    baseline_modalities = {ev.modality for ev in baseline_hits}
    
    print(f"Baseline Text-Only Hits ({len(baseline_hits)} items):")
    for ev in baseline_hits:
        print(f"  - [{ev.modality.upper()}] {ev.id} @ {ev.timestamp}s: '{ev.content}'")
    assert baseline_modalities == {"audio"}, "Baseline text-only RAG should only contain audio transcript text"
    
    # ContextMesh: Seed Search + Relationship Graph Expansion
    mm_seeds = search_evidence(query, all_evidence, top_k=2)
    mm_expanded = expand_relationships(mm_seeds, all_evidence, max_hops=1)
    mm_modalities = {ev.modality for ev in mm_expanded}
    
    print(f"\nContextMesh Multimodal Hits ({len(mm_expanded)} items):")
    for ev in mm_expanded:
        loc = f"@{ev.timestamp}s" if ev.timestamp is not None else f"Page {ev.page}" if ev.page is not None else "Image"
        print(f"  - [{ev.modality.upper()}] {ev.id} ({loc}): '{ev.content[:55]}...'")
        
    assert "video_frame" in mm_modalities, "ContextMesh must recover visual video frame evidence"
    assert "pdf" in mm_modalities, "ContextMesh must recover PDF specification evidence"
    print("\n Baseline comparison verified: ContextMesh surfaces cross-modal evidence absent in text-only RAG!")


def test_insufficient_evidence_behavior():
    """
    Validates that an unrelated/insufficient query returns 0 hits
    and produces an explicit GroundedAnswer stating evidence is insufficient.
    """
    print("\n--- Evaluation Test 3: Insufficient Evidence Behavior ---")
    all_evidence = load_evidence()
    unrelated_query = "What is the capital of France?"
    
    seeds = search_evidence(unrelated_query, all_evidence, top_k=2)
    assert len(seeds) == 0, f"Unrelated query should yield 0 direct hits, got: {seeds}"
    
    expanded = expand_relationships(seeds, all_evidence, max_hops=1)
    assert len(expanded) == 0, f"Unrelated query expansion should yield 0 items, got: {expanded}"
    
    from app import ask
    answer_obj = ask(unrelated_query, verbose=False)
    assert isinstance(answer_obj, GroundedAnswer)
    assert len(answer_obj.cited_evidence) == 0
    assert "insufficient" in answer_obj.answer.lower()
    print(" Insufficient evidence behavior verified: Correctly returns 0 hits and reports insufficient evidence!")


def run_evaluation_suite():
    print("=" * 65)
    print("Running Multimodal Retrieval & Evaluation Contract (Person 2)")
    print("=" * 65)
    
    test_retrieval_and_relationship_expansion_contract()
    test_baseline_vs_multimodal_rag_comparison()
    test_insufficient_evidence_behavior()
    
    print("\n" + "=" * 65)
    print(" ALL EVALUATION CONTRACT TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation_suite()
