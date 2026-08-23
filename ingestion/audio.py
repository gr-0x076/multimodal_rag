import os
import json
import re
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

# Ensure ffmpeg binary is on PATH for whisper
try:
    import imageio_ffmpeg, shutil
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not os.path.exists(target_ffmpeg) and os.path.exists(ffmpeg_exe):
        try:
            shutil.copyfile(ffmpeg_exe, target_ffmpeg)
        except Exception:
            pass
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# Ensure workspace root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence


def extract_entities(text: str) -> List[str]:
    """Extract meaningful technical terms and proper nouns, excluding English stopwords and system boilerplate."""
    # Technical domain keywords — always include when found
    tech_keywords = {
        "redis", "caching", "cache", "database", "sql", "nosql", "python", "api",
        "contextmesh", "multimodal", "rag", "architecture", "system", "function",
        "latency", "load", "server", "cluster", "node", "index", "vector",
        "def", "class", "return", "import", "loop", "recursion", "algorithm",
        "variable", "parameter", "argument", "indentation", "keyword", "syntax",
        "print", "statement", "method", "object", "list", "dict", "tuple",
        "machine", "learning", "neural", "network", "model", "training", "data",
        "code", "program", "script", "module", "library", "package", "pip",
    }

    # System/OCR boilerplate — always exclude
    boilerplate = {
        "image", "jpeg", "jpg", "png", "ocr", "path", "readme",
        "dimensions", "format", "unavailable", "detected", "extracted",
        "video", "frame", "file", "information",
    }

    # Common English stopwords and filler words (sentence starters, articles, conjunctions etc.)
    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "we", "our", "you",
        "your", "he", "she", "they", "them", "his", "her", "their", "my",
        "i", "me", "us", "who", "which", "what", "where", "when", "how",
        "why", "so", "as", "not", "no", "yes", "all", "any", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "than", "then",
        "there", "here", "now", "just", "also", "only", "very", "still",
        "about", "after", "again", "also", "because", "before", "between",
        "during", "into", "like", "over", "same", "see", "since", "through",
        "under", "until", "up", "use", "using", "used", "say", "said", "know",
        "think", "look", "want", "give", "go", "come", "make", "take", "get",
        "let", "put", "out", "down", "off", "away", "back", "even", "well",
        "way", "lot", "thing", "things", "time", "today", "new", "one", "two",
        "three", "first", "second", "next", "last", "right", "left", "good",
        "big", "little", "old", "long", "great", "small", "own", "part",
        "place", "case", "point", "number", "group", "problem", "example",
        # Sentence-starter capitals that are not proper nouns
        "now", "look", "the", "and", "but", "so", "here", "let", "see",
    }

    found = set()
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*\b', text)

    for word in words:
        w_lower = word.lower()

        # Skip boilerplate and stopwords
        if w_lower in boilerplate or w_lower in stopwords:
            continue

        # Always include known tech keywords (normalised)
        if w_lower in tech_keywords:
            # Preserve capitalisation for well-known proper tech names
            caps_map = {"redis": "Redis", "python": "Python", "sql": "SQL",
                        "nosql": "NoSQL", "api": "API", "rag": "RAG", "def": "def"}
            found.add(caps_map.get(w_lower, w_lower))
            continue

        # Include single-word proper nouns (capitalised, length > 2, not all-caps abbreviations)
        if word[0].isupper() and len(word) > 2 and not word.isupper():
            found.add(word)

    return sorted(list(found))



def transcribe_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_size: str = "base"
) -> List[Dict[str, Any]]:
    """
    Transcribes a video or audio file using OpenAI Whisper.
    Returns a list of timestamped segment dicts.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    print(f"Loading Whisper model ('{model_size}')...")
    import whisper
    model = whisper.load_model(model_size)

    print(f"Transcribing video: {video_path}...")
    result = model.transcribe(video_path, word_timestamps=False)

    segments = []
    for segment in result.get("segments", []):
        text_clean = segment["text"].strip()
        # Ignore single generic hallucinated tokens like "you" or empty strings
        if not text_clean or text_clean.lower() in {"you", "you.", "thank you.", "subtitles by"} or len(text_clean) < 3:
            continue
        segments.append({
            "start": round(float(segment["start"]), 2),
            "end": round(float(segment["end"]), 2),
            "text": text_clean
        })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        print(f"Transcript saved to {output_path}")

    return segments


def extract_audio_evidence(
    video_path: str,
    output_transcript_path: Optional[str] = None,
    source_name: Optional[str] = None,
    model_size: str = "base"
) -> List[Evidence]:
    """
    Transcribes audio from a video file and converts segments into Evidence objects.
    """
    source = source_name or os.path.basename(video_path)
    source_clean = Path(source).stem.lower()

    segments = transcribe_video(
        video_path,
        output_path=output_transcript_path,
        model_size=model_size
    )

    evidence_list = []
    for i, seg in enumerate(segments):
        start_time = seg["start"]
        end_time = seg["end"]
        text = seg["text"]
        entities = extract_entities(text)

        ev_id = f"{source_clean}_audio_{i}"
        ev = Evidence(
            id=ev_id,
            content=text,
            modality="audio",
            source=source,
            timestamp=start_time,
            entities=entities,
            confidence=0.95,
            relationships=[],
            metadata={
                "start": start_time,
                "end": end_time,
                "duration": round(end_time - start_time, 2)
            }
        )
        evidence_list.append(ev)

    return evidence_list


if __name__ == "__main__":
    import sys
    test_video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/meeting.mp4"
    if os.path.exists(test_video):
        evidences = extract_audio_evidence(test_video, "data/processed/transcript.json")
        print(f"Extracted {len(evidences)} audio evidence items.")
    else:
        print(f"File not found: {test_video}. Please place a test video in data/raw/meeting.mp4.")
