"""Debug script to verify Whisper transcription on my_test_video."""
import whisper
import imageio_ffmpeg
import os

# Patch ffmpeg PATH
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

model = whisper.load_model("base")

print("=" * 60)
print("Transcribing: data/raw/my_test_video.mp4")
print("=" * 60)
result = model.transcribe("data/raw/my_test_video.mp4", word_timestamps=False)
print(f"Language detected: {result.get('language', '?')}")
print(f"Full text: {repr(result.get('text', '')[:500])}")
print(f"Raw segments: {len(result.get('segments', []))}")
for seg in result.get("segments", [])[:10]:
    t = seg["text"]
    print(f"  [{seg['start']:.1f}s-{seg['end']:.1f}s]: {repr(t)}")
