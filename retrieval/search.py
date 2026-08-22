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
    if not all_evidence:
        return []
        
    query_tokens = tokenize(query)
    scored: List[Tuple[float, Evidence]] = []
    
    for ev in all_evidence:
        content_tokens = tokenize(ev.content)
        entity_tokens = set(e.lower() for e in ev.entities)
        
        content_matches = len(query_tokens.intersection(content_tokens))
        entity_matches = len(query_tokens.intersection(entity_tokens))
        
        # Scoring: Entity matches have higher semantic importance
        score = (content_matches * 1.0) + (entity_matches * 2.5)
        
        if score > 0:
            scored.append((score, ev))
            
    # Sort by relevance score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]
