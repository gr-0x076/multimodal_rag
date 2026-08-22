from typing import List, Dict, Set
from knowledge.schema import Evidence


def expand_relationships(
    seed_evidence: List[Evidence],
    all_evidence: List[Evidence],
    max_hops: int = 1
) -> List[Evidence]:
    """
    Standard Relationship Expansion Contract.
    Expands seed evidence items by traversing their 1-hop connected graph edges.
    
    Args:
        seed_evidence: The initial high-relevance evidence objects from search.
        all_evidence: The complete collection of all evidence objects (or index).
        max_hops: Number of relationship hops to traverse (default: 1).
        
    Returns:
        Deduplicated list of Evidence objects containing the seeds and their connected nodes.
    """
    evidence_by_id: Dict[str, Evidence] = {ev.id: ev for ev in all_evidence}
    visited_ids: Set[str] = set()
    result: List[Evidence] = []
    
    # 1. Add direct seed hits
    current_tier = []
    for ev in seed_evidence:
        if ev.id not in visited_ids:
            visited_ids.add(ev.id)
            result.append(ev)
            current_tier.append(ev)
            
    # 2. Traverse up to max_hops
    for _ in range(max_hops):
        next_tier = []
        for ev in current_tier:
            for rel_id in ev.relationships:
                if rel_id in evidence_by_id and rel_id not in visited_ids:
                    visited_ids.add(rel_id)
                    target_ev = evidence_by_id[rel_id]
                    result.append(target_ev)
                    next_tier.append(target_ev)
        current_tier = next_tier
        if not current_tier:
            break
            
    return result
    
    
def expand_related(
    seed_evidence: List[Evidence],
    all_evidence: List[Evidence],
    depth: int = 1
) -> List[Evidence]:
    """Alias for expand_relationships supporting legacy parameter name (depth -> max_hops)."""
    return expand_relationships(seed_evidence, all_evidence, max_hops=depth)

