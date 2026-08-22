"""
ContextMesh — Multimodal Retrieval Engine

Accepts a natural-language query and returns the most relevant Evidence
objects across all modalities (audio, video_frame, pdf, image).

Scoring uses TF-IDF-style keyword relevance — lightweight, no model
downloads required.
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from knowledge.relationships import expand_related


DEFAULT_EVIDENCE_PATH = os.path.join(root_dir, "data", "processed", "evidence.json")


# ---------------------------------------------------------------------------
# Evidence Loading
# ---------------------------------------------------------------------------

def load_evidence(path: Optional[str] = None) -> List[Evidence]:
    """
    Load Evidence objects from a JSON file.
    Returns an empty list if the file doesn't exist.
    """
    path = path or DEFAULT_EVIDENCE_PATH
    if not os.path.exists(path):
        print(f"[retrieval] Evidence file not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
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
            metadata=item.get("metadata", {}),
        ))
    return evidence_list


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization, strip punctuation."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


def _idf(term: str, doc_tokens: List[List[str]]) -> float:
    """Inverse document frequency."""
    n = len(doc_tokens)
    df = sum(1 for tokens in doc_tokens if term in tokens)
    if df == 0:
        return 0.0
    return math.log((n + 1) / (df + 1)) + 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_evidence(
    query: str,
    evidence_list: List[Evidence],
) -> List[Tuple[float, Evidence]]:
    """
    Score every Evidence object against the query using TF-IDF-style
    relevance.  Returns (score, evidence) pairs sorted by descending score.

    Searches across ALL modalities — content text AND entity lists.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Build per-document token lists (content + entities combined)
    doc_tokens: List[List[str]] = []
    for ev in evidence_list:
        combined = ev.content + " " + " ".join(ev.entities)
        doc_tokens.append(_tokenize(combined))

    # Pre-compute IDF
    unique_query_terms = set(query_tokens)
    idf_cache: Dict[str, float] = {
        term: _idf(term, doc_tokens) for term in unique_query_terms
    }

    query_tf = Counter(query_tokens)

    scored: List[Tuple[float, Evidence]] = []
    for idx, ev in enumerate(evidence_list):
        tokens = doc_tokens[idx]
        if not tokens:
            continue

        doc_tf = Counter(tokens)
        score = 0.0
        for term in unique_query_terms:
            tf_q = query_tf[term]
            tf_d = doc_tf.get(term, 0)
            if tf_d > 0:
                score += tf_q * (1 + math.log(tf_d)) * idf_cache[term]

        # Boost by evidence confidence
        score *= ev.confidence

        if score > 0:
            scored.append((score, ev))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Main Search API
# ---------------------------------------------------------------------------

def search(
    query: str,
    evidence_path: Optional[str] = None,
    top_k: int = 10,
    expand_depth: int = 1,
) -> List[Evidence]:
    """
    End-to-end retrieval pipeline:

    1. Load evidence from JSON.
    2. Score & rank by relevance to the query.
    3. Take top-k direct hits.
    4. Expand via relationship links.
    5. Deduplicate & return.

    Usage:
        results = search("What architecture was proposed to reduce database load?")
    """
    all_evidence = load_evidence(evidence_path)
    if not all_evidence:
        return []

    scored = score_evidence(query, all_evidence)
    direct_hits = [ev for _, ev in scored[:top_k]]

    if not direct_hits:
        return []

    # Expand through relationship graph
    expanded = expand_related(direct_hits, all_evidence, depth=expand_depth)
    return expanded


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "What architecture was proposed to reduce database load?"
    print(f"\nQuery: {query}")
    print("=" * 60)

    results = search(query)

    if not results:
        print("No results found. Make sure data/processed/evidence.json exists.")
        print("Run: python pipeline/ingest.py  OR  python -m tests.create_seed_data")
    else:
        print(f"Found {len(results)} evidence items:\n")
        for ev in results:
            print(f"  [{ev.modality.upper():12s}] {ev.id}")
            print(f"    Content : {ev.content[:80]}..." if len(ev.content) > 80 else f"    Content : {ev.content}")
            print(f"    Source  : {ev.source}", end="")
            if ev.timestamp is not None:
                mins, secs = divmod(int(ev.timestamp), 60)
                print(f" @ {mins:02d}:{secs:02d}", end="")
            if ev.page is not None:
                print(f" -- page {ev.page}", end="")
            print(f"\n    Related : {ev.relationships}")
            print()
