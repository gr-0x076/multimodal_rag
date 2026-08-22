"""
ContextMesh — Relationship Engine

Extracts and manages relationships between Evidence objects:
  - Temporal: evidence occurring close in time (same source)
  - Entity: evidence sharing key entities/concepts
  - Source: evidence from the same source file

No external graph DB required — relationships are stored as ID lists
on each Evidence object and serialized to JSON.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence


# ---------------------------------------------------------------------------
# 1. Temporal Relationships
# ---------------------------------------------------------------------------

def find_temporal_relations(
    evidence_list: List[Evidence],
    window_seconds: float = 6.0,
) -> List[Tuple[str, str]]:
    """
    Link evidence objects whose timestamps fall within *window_seconds*
    of each other **and** originate from the same source file.

    Returns a list of (id_a, id_b) pairs.
    """
    pairs: List[Tuple[str, str]] = []
    timed = [ev for ev in evidence_list if ev.timestamp is not None]

    for i, ev_a in enumerate(timed):
        for ev_b in timed[i + 1:]:
            if ev_a.source != ev_b.source:
                continue
            if ev_a.id == ev_b.id:
                continue
            if abs(ev_a.timestamp - ev_b.timestamp) <= window_seconds:
                pairs.append((ev_a.id, ev_b.id))
    return pairs


# ---------------------------------------------------------------------------
# 2. Entity Relationships
# ---------------------------------------------------------------------------

def find_entity_relations(
    evidence_list: List[Evidence],
    min_overlap: int = 1,
) -> List[Tuple[str, str]]:
    """
    Link evidence objects that share at least *min_overlap* entities
    (case-insensitive comparison).  Works across all modalities.

    Returns a list of (id_a, id_b) pairs.
    """
    pairs: List[Tuple[str, str]] = []

    entity_sets: Dict[str, Set[str]] = {}
    for ev in evidence_list:
        entity_sets[ev.id] = {e.lower() for e in ev.entities} if ev.entities else set()

    ids = list(entity_sets.keys())
    for i, id_a in enumerate(ids):
        set_a = entity_sets[id_a]
        if not set_a:
            continue
        for id_b in ids[i + 1:]:
            set_b = entity_sets[id_b]
            if not set_b:
                continue
            if len(set_a & set_b) >= min_overlap:
                pairs.append((id_a, id_b))
    return pairs


# ---------------------------------------------------------------------------
# 3. Source Relationships
# ---------------------------------------------------------------------------

def find_source_relations(
    evidence_list: List[Evidence],
) -> List[Tuple[str, str]]:
    """
    Link evidence objects that originate from the same source file.

    Returns a list of (id_a, id_b) pairs.
    """
    by_source: Dict[str, List[str]] = defaultdict(list)
    for ev in evidence_list:
        by_source[ev.source].append(ev.id)

    pairs: List[Tuple[str, str]] = []
    for source, ids in by_source.items():
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                pairs.append((id_a, id_b))
    return pairs


# ---------------------------------------------------------------------------
# 4. Graph Builder — merge all relation types
# ---------------------------------------------------------------------------

def _apply_pairs(
    evidence_by_id: Dict[str, Evidence],
    pairs: List[Tuple[str, str]],
) -> None:
    """Write relationship pairs back onto Evidence objects (bidirectional)."""
    for id_a, id_b in pairs:
        ev_a = evidence_by_id.get(id_a)
        ev_b = evidence_by_id.get(id_b)
        if ev_a is None or ev_b is None:
            continue
        if id_b not in ev_a.relationships:
            ev_a.relationships.append(id_b)
        if id_a not in ev_b.relationships:
            ev_b.relationships.append(id_a)


def build_relationship_graph(
    evidence_list: List[Evidence],
    temporal_window: float = 6.0,
    entity_min_overlap: int = 1,
) -> List[Evidence]:
    """
    Run all relationship extractors and write the results back
    into each Evidence object's ``relationships`` list.

    Returns the same list (mutated in-place) for convenience.
    """
    evidence_by_id = {ev.id: ev for ev in evidence_list}

    temporal_pairs = find_temporal_relations(evidence_list, temporal_window)
    entity_pairs = find_entity_relations(evidence_list, entity_min_overlap)
    source_pairs = find_source_relations(evidence_list)

    _apply_pairs(evidence_by_id, temporal_pairs)
    _apply_pairs(evidence_by_id, entity_pairs)
    _apply_pairs(evidence_by_id, source_pairs)

    return evidence_list


# ---------------------------------------------------------------------------
# 5. Relationship Expansion (for retrieval)
# ---------------------------------------------------------------------------

def expand_related(
    seed_evidence: List[Evidence],
    all_evidence: List[Evidence],
    depth: int = 1,
) -> List[Evidence]:
    """
    Starting from *seed_evidence*, follow relationship links up to *depth*
    hops and return the expanded set (deduplicated, seed included).
    """
    evidence_by_id = {ev.id: ev for ev in all_evidence}
    seen: Set[str] = set()
    result: List[Evidence] = []

    # Start with seed
    frontier = list(seed_evidence)

    for _ in range(depth + 1):  # depth 0 = seed only
        next_frontier: List[Evidence] = []
        for ev in frontier:
            if ev.id in seen:
                continue
            seen.add(ev.id)
            result.append(ev)
            # Queue neighbours for next hop
            for rel_id in ev.relationships:
                if rel_id not in seen:
                    related = evidence_by_id.get(rel_id)
                    if related:
                        next_frontier.append(related)
        frontier = next_frontier

    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = [
        Evidence(
            id="audio_1", content="Use Redis to reduce database load",
            modality="audio", source="meeting.mp4", timestamp=10.0,
            entities=["Redis", "database", "load"],
        ),
        Evidence(
            id="frame_1", content="Redis architecture diagram",
            modality="video_frame", source="meeting.mp4", timestamp=12.0,
            entities=["Redis", "architecture"],
        ),
        Evidence(
            id="pdf_1", content="Redis is an in-memory caching layer for databases",
            modality="pdf", source="architecture.pdf", page=2,
            entities=["Redis", "caching", "database"],
        ),
    ]

    build_relationship_graph(demo)

    print("\n=== Relationship Graph ===")
    for ev in demo:
        print(f"  {ev.id} ({ev.modality}) -> {ev.relationships}")

    print("\n=== Expansion from audio_1 ===")
    expanded = expand_related([demo[0]], demo, depth=1)
    for ev in expanded:
        print(f"  {ev.id} ({ev.modality}): {ev.content}")
