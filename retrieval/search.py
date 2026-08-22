import re
from typing import List, Tuple
from knowledge.schema import Evidence


STOPWORDS = {
    "what", "is", "the", "of", "and", "to", "in", "a", "an", "where", "how",
    "why", "which", "who", "for", "with", "on", "at", "by", "from", "it",
    "this", "that", "was", "were", "are", "be", "been", "being", "have", "has", "had"
}


def tokenize(text: str) -> set:
    """Helper to tokenize strings into normalized lowercase tokens (excluding stopwords)."""
    tokens = set(re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()))
    return tokens - STOPWORDS


def score_evidence(query: str, all_evidence: List[Evidence]) -> List[Tuple[float, Evidence]]:
    """
    Score evidence objects against a query string.
    Returns list of (score, evidence) tuples sorted by descending score.
    """
    if not all_evidence:
        return []
        
    query_tokens = tokenize(query)
    scored: List[Tuple[float, Evidence]] = []
    
    for ev in all_evidence:
        content_tokens = tokenize(ev.content)
        entity_tokens = set(e.lower() for e in ev.entities)
        
        content_matches = len(query_tokens.intersection(content_tokens))
        entity_matches = len(query_tokens.intersection(entity_tokens))
        
        score = (content_matches * 1.0) + (entity_matches * 2.5)
        if score > 0:
            scored.append((score, ev))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def search_evidence(
    query: str,
    all_evidence: List[Evidence],
    top_k: int = 3
) -> List[Evidence]:
    """
    Standard Retrieval Contract for ContextMesh.
    
    Args:
        query: User query string (e.g. 'What architecture was proposed to reduce database load?')
        all_evidence: Collection of candidate Evidence objects.
        top_k: Number of highest ranking direct evidence hits to return.
        
    Returns:
        List[Evidence]: Top-k ranked evidence items based on content & entity relevance.
    """
    scored = score_evidence(query, all_evidence)
    return [item[1] for item in scored[:top_k]]


def search(query: str, evidence_path: str = None, all_evidence: List[Evidence] = None, top_k: int = 3, expand_depth: int = 0) -> List[Evidence]:
    """Alias for search_evidence supporting legacy/test calling conventions."""
    if all_evidence is None:
        import json
        import os
        from pathlib import Path
        root = str(Path(__file__).resolve().parent.parent)
        path = evidence_path or os.path.join(root, "data", "processed", "evidence.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_evidence = [Evidence(**item) for item in data]
        else:
            all_evidence = []
    
    seeds = search_evidence(query, all_evidence, top_k=top_k)
    if expand_depth > 0:
        from knowledge.relationships import expand_relationships
        return expand_relationships(seeds, all_evidence, max_hops=expand_depth)
    return seeds

