"""
ContextMesh — Retrieval & Grounding Unit Tests

Validates:
  1. TF-IDF scoring correctness
  2. Relationship expansion and deduplication
  3. Grounded context formatting
"""

import sys
import unittest
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from retrieval.search import score_evidence, search
from retrieval.grounding import format_evidence_context, build_grounded_prompt


class TestRetrievalGrounding(unittest.TestCase):
    def setUp(self):
        # Create a mock database of evidence
        self.evidence = [
            Evidence(
                id="audio_redis",
                content="We should use Redis caching to reduce database load.",
                modality="audio",
                source="meeting.mp4",
                timestamp=10.0,
                entities=["Redis", "caching", "database", "load"],
                relationships=["frame_redis", "pdf_caching"],
            ),
            Evidence(
                id="frame_redis",
                content="Redis architecture slide showing the caching layer.",
                modality="video_frame",
                source="meeting.mp4",
                timestamp=12.0,
                entities=["Redis", "caching"],
                relationships=["audio_redis"],
            ),
            Evidence(
                id="pdf_caching",
                content="Caching is defined as storing database queries in-memory for speed.",
                modality="pdf",
                source="architecture.pdf",
                page=2,
                entities=["caching", "database"],
                relationships=["audio_redis"],
            ),
            Evidence(
                id="image_other",
                content="Whiteboard drawing of user authentication flow.",
                modality="image",
                source="auth.png",
                entities=["auth", "user"],
                relationships=[],
            )
        ]

    def test_relevance_scoring(self):
        # Searching for "Redis load" should rank audio_redis and frame_redis high, but not image_other
        scored = score_evidence("Redis load", self.evidence)
        self.assertTrue(len(scored) > 0)
        
        top_score, top_ev = scored[0]
        self.assertEqual(top_ev.id, "audio_redis")

        # Verify image_other has score 0 (or is not in scored)
        scored_ids = [ev.id for _, ev in scored]
        self.assertNotIn("image_other", scored_ids)

    def test_relationship_expansion(self):
        # If we search for "Redis load" with top_k=1, we get audio_redis.
        # But we expect the search to expand to frame_redis and pdf_caching (since they are related).
        results = search("Redis load", evidence_path="dummy_path", top_k=1, expand_depth=1)
        
        # Override mock loading since search normally reads from file
        # We'll test expand_related directly or mock load_evidence
        from knowledge.relationships import expand_related
        direct_hits = [self.evidence[0]]  # audio_redis
        expanded = expand_related(direct_hits, self.evidence, depth=1)
        
        expanded_ids = [ev.id for ev in expanded]
        self.assertIn("audio_redis", expanded_ids)
        self.assertIn("frame_redis", expanded_ids)
        self.assertIn("pdf_caching", expanded_ids)
        self.assertNotIn("image_other", expanded_ids)

    def test_grounding_formatting(self):
        context = format_evidence_context(self.evidence[:3])
        self.assertIn("ID       : audio_redis", context)
        self.assertIn("Timestamp: 00:10 (10.0s)", context)
        self.assertIn("Page     : 2", context)
        self.assertIn("Related  : frame_redis, pdf_caching", context)

    def test_grounded_prompt_rules(self):
        system_prompt, user_prompt = build_grounded_prompt("How do we reduce load?", self.evidence[:3])
        self.assertIn("You are ContextMesh, a multimodal evidence-grounded assistant.", system_prompt)
        self.assertIn("Answer the question using ONLY the evidence above.", user_prompt)
        self.assertIn("Cite sources with their timestamps or page numbers.", user_prompt)


if __name__ == "__main__":
    unittest.main()
