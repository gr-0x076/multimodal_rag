import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from pipeline.ingest import ingest_all


EVIDENCE_PATH = os.path.join(root_dir, "data", "processed", "evidence.json")


def load_evidence() -> List[Evidence]:
    """Ensures ingestion has run and loads evidence objects from evidence.json."""
    if not os.path.exists(EVIDENCE_PATH):
        print(f"evidence.json not found at {EVIDENCE_PATH}. Running ingestion pipeline...")
        ingest_all(data_dir=os.path.join(root_dir, "data"), processed_dir=os.path.join(root_dir, "data", "processed"))
    
    with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    
    evidence_list = []
    for item in raw_items:
        evidence_list.append(Evidence(
            id=item["id"],
            content=item["content"],
            modality=item["modality"],
            source=item["source"],
            timestamp=item.get("timestamp"),
            page=item.get("page"),
            entities=item.get("entities", []),
            confidence=item.get("confidence", 1.0),
            relationships=item.get("relationships", []),
            metadata=item.get("metadata", {})
        ))
    return evidence_list


def test_unified_evidence_loading():
    """Validates that evidence.json loads successfully into typed Evidence dataclasses."""
    evidence = load_evidence()
    assert len(evidence) > 0, "Evidence collection should not be empty"
    print(f" Loaded {len(evidence)} total evidence objects.")


def test_all_four_modalities_present():
    """Validates that all four multimodal sources (Audio, Video Frame, PDF, Image) coexist in the unified store."""
    evidence = load_evidence()
    modalities = {item.modality for item in evidence}
    print(f"Detected modalities: {modalities}")
    
    required_modalities = {"audio", "video_frame", "pdf", "image"}
    for mod in required_modalities:
        assert mod in modalities, f"Missing required modality: '{mod}' in evidence store!"
    print(" All 4 required modalities ('audio', 'video_frame', 'pdf', 'image') are present.")


def test_provenance_and_schema_integrity():
    """Validates that all evidence objects adhere strictly to the Evidence contract and retain provenance."""
    evidence = load_evidence()
    for ev in evidence:
        assert ev.id, f"Evidence missing ID: {ev}"
        assert ev.content, f"Evidence missing content: {ev.id}"
        assert ev.modality in {"audio", "video_frame", "pdf", "image", "text"}, f"Invalid modality in {ev.id}"
        assert ev.source, f"Evidence missing source filename: {ev.id}"
        assert 0.0 <= ev.confidence <= 1.0, f"Confidence out of bounds in {ev.id}: {ev.confidence}"
        
        # Modality-specific provenance checks
        if ev.modality in {"audio", "video_frame"}:
            assert ev.timestamp is not None, f"Audio/Video evidence {ev.id} must have a timestamp"
        if ev.modality == "pdf":
            assert ev.page is not None, f"PDF evidence {ev.id} must have a page number"
            
    print(" Provenance and schema integrity verified for all items.")


def test_cross_modal_and_temporal_relationships():
    """Validates that relationships exist between temporal events and cross-modal entity mentions."""
    evidence = load_evidence()
    evidence_by_id = {ev.id: ev for ev in evidence}
    
    has_temporal_link = False
    has_cross_modal_entity_link = False
    
    for ev in evidence:
        for rel_id in ev.relationships:
            assert rel_id in evidence_by_id, f"Dangling relationship pointer: {ev.id} -> {rel_id}"
            target_ev = evidence_by_id[rel_id]
            
            # Check for temporal link between audio and frame
            if (ev.modality == "audio" and target_ev.modality == "video_frame") or \
               (ev.modality == "video_frame" and target_ev.modality == "audio"):
                if ev.timestamp is not None and target_ev.timestamp is not None:
                    if abs(ev.timestamp - target_ev.timestamp) <= 6.0:
                        has_temporal_link = True
            
            # Check for cross-modal link between audio and PDF/Image
            if (ev.modality == "audio" and target_ev.modality in {"pdf", "image"}) or \
               (ev.modality in {"pdf", "image"} and target_ev.modality == "audio"):
                has_cross_modal_entity_link = True
                
    assert has_temporal_link, "Expected at least one temporal relationship between Audio and Video Frame"
    assert has_cross_modal_entity_link, "Expected at least one cross-modal entity relationship between Audio and PDF/Image"
    print(" Verified both temporal links and cross-modal entity relationships.")


def test_golden_query_evidence_chain():
    """
    Validates the Hero Demo / Golden Query scenario:
    Query: 'What architecture was proposed to reduce database load, and what visual/document evidence supports it?'
    Expected Chain:
      - Audio discussing Redis caching to reduce database load (~10s)
      - Connected Video Frame at ~10s showing the Redis architecture visual
      - Connected PDF Document Page 2 with the technical caching specification
    """
    print("\n--- Testing Golden Query Evidence Chain ---")
    evidence = load_evidence()
    evidence_by_id = {ev.id: ev for ev in evidence}
    
    # 1. Locate primary audio trigger mentioning Redis & database load
    audio_redis_nodes = [
        ev for ev in evidence 
        if ev.modality == "audio" and ("redis" in ev.content.lower() or "redis" in [e.lower() for e in ev.entities])
    ]
    assert len(audio_redis_nodes) > 0, "Golden Query: Failed to find audio node mentioning Redis"
    primary_audio = audio_redis_nodes[0]
    print(f"1. Primary Audio Found: [{primary_audio.id}] @ {primary_audio.timestamp}s: '{primary_audio.content}'")
    
    # 2. Traverse relationships to find connected video frame
    connected_frames = [
        evidence_by_id[rel_id] for rel_id in primary_audio.relationships 
        if rel_id in evidence_by_id and evidence_by_id[rel_id].modality == "video_frame"
    ]
    assert len(connected_frames) > 0, "Golden Query: Failed to find connected Video Frame evidence"
    primary_frame = connected_frames[0]
    print(f"2. Connected Video Frame: [{primary_frame.id}] @ {primary_frame.timestamp}s (Source: {primary_frame.source})")
    
    # 3. Traverse relationships to find connected PDF specification
    connected_pdfs = [
        evidence_by_id[rel_id] for rel_id in primary_audio.relationships 
        if rel_id in evidence_by_id and evidence_by_id[rel_id].modality == "pdf"
    ]
    assert len(connected_pdfs) > 0, "Golden Query: Failed to find connected PDF document evidence"
    primary_pdf = connected_pdfs[0]
    print(f"3. Connected PDF Evidence: [{primary_pdf.id}] Page {primary_pdf.page}: '{primary_pdf.content[:60]}...'")
    
    print("\n Golden Query evidence chain successfully verified across Audio -> Frame -> PDF!")


def run_all_integration_tests():
    print("=" * 60)
    print("Running Milestone 2 End-to-End Integration Tests (Person 2)")
    print("=" * 60)
    
    test_unified_evidence_loading()
    test_all_four_modalities_present()
    test_provenance_and_schema_integrity()
    test_cross_modal_and_temporal_relationships()
    test_golden_query_evidence_chain()
    
    print("\n" + "=" * 60)
    print(" ALL END-TO-END INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_integration_tests()
