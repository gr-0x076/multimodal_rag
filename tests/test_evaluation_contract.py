import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from tests.test_end_to_end import load_evidence


def expand_relationships(
    seed_evidence: List[Evidence],
    evidence_by_id: Dict[str, Evidence],
    max_hops: int = 1
) -> List[Evidence]:
    """
    Expands a list of seed evidence objects by traversing their 1-hop relationship graph.
    Returns the deduplicated expanded evidence collection preserving provenance.
    """
    expanded_set: Set[str] = set()
    result: List[Evidence] = []
    
    # 1. Add direct seed hits
    for ev in seed_evidence:
        if ev.id not in expanded_set:
            expanded_set.add(ev.id)
            result.append(ev)
            
    # 2. Traverse 1-hop relationships
    for ev in seed_evidence:
        for rel_id in ev.relationships:
            if rel_id in evidence_by_id and rel_id not in expanded_set:
                expanded_set.add(rel_id)
                result.append(evidence_by_id[rel_id])
                
    return result


def simple_keyword_search(
    query: str,
    all_evidence: List[Evidence],
    top_k: int = 3
) -> List[Evidence]:
    """
    Deterministic reference retriever based on term overlap and entity matching.
    Serves as the contract baseline before vector/Chroma retrieval is plugged in.
    """
    query_terms = set(query.lower().replace("?", "").replace(",", "").split())
    scored: List[Tuple[float, Evidence]] = []
    
    for ev in all_evidence:
        score = 0.0
        content_words = set(ev.content.lower().split())
        entity_words = set(e.lower() for e in ev.entities)
        
        # Overlap with content
        content_overlap = len(query_terms.intersection(content_words))
        score += content_overlap * 1.5
        
        # Overlap with extracted entities (higher weight)
        entity_overlap = len(query_terms.intersection(entity_words))
        score += entity_overlap * 3.0
        
        if score > 0:
            scored.append((score, ev))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def test_golden_query_retrieval_and_expansion():
    """
    Tests the deterministic evaluation contract for the Golden Query:
    Query: 'What architecture was proposed to reduce database load, and what visual/document evidence supports it?'
    
    Validates that:
      1. Direct retriever finds relevant spoken/written evidence (audio & PDF).
      2. Graph expansion discovers the corresponding visual video frame at the same timestamp.
    """
    print("\n--- Evaluation Test 1: Golden Query Deterministic Retrieval ---")
    evidence_list = load_evidence()
    evidence_by_id = {ev.id: ev for ev in evidence_list}
    
    query = "What architecture was proposed to reduce database load?"
    
    # 1. Direct Retrieval
    top_hits = simple_keyword_search(query, evidence_list, top_k=3)
    top_hit_ids = {ev.id for ev in top_hits}
    print(f"Top direct retrieval hits ({len(top_hits)}): {top_hit_ids}")
    
    assert "meeting_audio_2" in top_hit_ids or "architecture_page_2" in top_hit_ids, \
        "Direct retrieval must capture either primary audio statement or PDF specification"
        
    # 2. Relationship Expansion (1-hop graph traversal)
    expanded_context = expand_relationships(top_hits, evidence_by_id, max_hops=1)
    expanded_ids = {ev.id for ev in expanded_context}
    print(f"Expanded multimodal context ({len(expanded_context)} items): {expanded_ids}")
    
    # Must contain Audio, PDF, and Video Frame
    assert "meeting_audio_2" in expanded_ids, "Expected meeting_audio_2 in grounded context"
    assert "architecture_page_2" in expanded_ids, "Expected architecture_page_2 in grounded context"
    
    frame_hits = [ev for ev in expanded_context if ev.modality == "video_frame"]
    assert len(frame_hits) > 0, "Graph expansion must bring in the temporally correlated video frame(s)"
    
    print(" Golden Query deterministic retrieval & relationship expansion passed!")


def test_baseline_vs_multimodal_rag_comparison():
    """
    Evaluates Text-Only Baseline RAG vs. ContextMesh Multimodal Graph RAG.
    This fulfills the hackathon requirement for comparative baseline evaluation.
    """
    print("\n--- Evaluation Test 2: Baseline vs Multimodal RAG Comparison ---")
    evidence_list = load_evidence()
    evidence_by_id = {ev.id: ev for ev in evidence_list}
    
    query = "What architecture was proposed to reduce database load?"
    
    # Baseline: Text transcript only
    transcript_evidence = [ev for ev in evidence_list if ev.modality == "audio"]
    baseline_hits = simple_keyword_search(query, transcript_evidence, top_k=2)
    baseline_modalities = {ev.modality for ev in baseline_hits}
    
    print(f"Baseline Text-Only Results ({len(baseline_hits)} items):")
    for ev in baseline_hits:
        print(f"  - [{ev.modality.upper()}] {ev.id} @ {ev.timestamp}s: '{ev.content}'")
        
    assert baseline_modalities == {"audio"}, "Baseline should only capture audio transcript text"
    
    # ContextMesh: Multimodal Seed + Relationship Expansion
    mm_seeds = simple_keyword_search(query, evidence_list, top_k=2)
    mm_expanded = expand_relationships(mm_seeds, evidence_by_id, max_hops=1)
    mm_modalities = {ev.modality for ev in mm_expanded}
    
    print(f"\nContextMesh Multimodal Results ({len(mm_expanded)} items):")
    for ev in mm_expanded:
        loc = f"@{ev.timestamp}s" if ev.timestamp is not None else f"Page {ev.page}" if ev.page is not None else "Image"
        print(f"  - [{ev.modality.upper()}] {ev.id} ({loc}): '{ev.content[:60]}...'")
        
    # Validation: Multimodal RAG recovers visual evidence & document specs that Baseline completely misses
    assert len(mm_modalities) >= 3, f"ContextMesh must span multiple modalities. Got: {mm_modalities}"
    assert "video_frame" in mm_modalities, "ContextMesh must recover visual video frame evidence"
    assert "pdf" in mm_modalities, "ContextMesh must recover PDF specification evidence"
    
    print("\n Comparative evaluation verified: ContextMesh recovers cross-modal evidence missed by text-only baseline!")


def run_evaluation_suite():
    print("=" * 65)
    print("Running Multimodal Retrieval & Evaluation Contract (Person 2)")
    print("=" * 65)
    
    test_golden_query_retrieval_and_expansion()
    test_baseline_vs_multimodal_rag_comparison()
    
    print("\n" + "=" * 65)
    print(" ALL EVALUATION CONTRACT TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation_suite()
