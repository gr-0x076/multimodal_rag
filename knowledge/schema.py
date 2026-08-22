from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Evidence:
    """
    Unified Evidence Object Schema for ContextMesh Multimodal RAG.
    All ingestion pipelines (Video, Audio, PDF, Image) produce Evidence objects conforming to this contract.
    """
    id: str
    content: str
    modality: str                       # 'pdf', 'image', 'audio', 'video_frame', 'text'
    source: str                         # Original file name or URL (e.g., 'architecture.pdf', 'meeting.mp4')
    timestamp: Optional[float] = None   # Timestamp in seconds (for audio/video/frames)
    page: Optional[int] = None          # Page number (for PDF / multi-page documents)
    entities: List[str] = field(default_factory=list)      # Extracted key entities (e.g. ['Redis', 'Database'])
    confidence: float = 1.0             # Confidence score (0.0 to 1.0)
    relationships: List[str] = field(default_factory=list) # IDs of connected/related Evidence objects
    metadata: Dict[str, Any] = field(default_factory=dict) # Any additional optional metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "modality": self.modality,
            "source": self.source,
            "timestamp": self.timestamp,
            "page": self.page,
            "entities": self.entities,
            "confidence": self.confidence,
            "relationships": self.relationships,
            "metadata": self.metadata,
        }


@dataclass
class GroundedAnswer:
    """
    Standard Answer contract for Person 3's LLM / Groq answering engine.
    Ensures answers are grounded in explicit multimodal evidence citations.
    """
    query: str
    answer: str
    cited_evidence: List[Evidence] = field(default_factory=list)
    modalities_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "cited_evidence": [ev.to_dict() for ev in self.cited_evidence],
            "modalities_used": self.modalities_used,
            "metadata": self.metadata,
        }
