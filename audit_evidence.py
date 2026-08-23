"""Full audit of data/processed/evidence.json."""
import json
import os

ev = json.load(open("data/processed/evidence.json", encoding="utf-8"))
print(f"Total nodes: {len(ev)}")
print()

modalities = {}
for e in ev:
    m = e["modality"]
    modalities.setdefault(m, []).append(e)

for mod, nodes in modalities.items():
    print(f"=== {mod.upper()} ({len(nodes)} nodes) ===")
    for n in nodes:
        nid = n["id"]
        ts = n.get("timestamp")
        pg = n.get("page")
        src = n.get("source", "?")
        content = n.get("content", "")[:90].replace("\n", " ")
        entities = n.get("entities", [])
        rels = len(n.get("relationships", []))
        meta = n.get("metadata", {})

        if mod == "audio":
            start = meta.get("start", ts)
            end = meta.get("end", "?")
            duration = meta.get("duration", "?")
            print(f"  [{nid}] source={src} @ {start}s-{end}s (dur={duration}s)")
            print(f"    Content: {repr(content)}")
            print(f"    Entities: {entities}")
            print(f"    Relationships: {rels}")

        elif mod == "video_frame":
            fp = meta.get("frame_path", "?")
            fp_exists = os.path.exists(fp)
            ocr = meta.get("ocr_text", "")
            print(f"  [{nid}] source={src} @ {ts}s | file_exists={fp_exists} | OCR={repr(ocr[:60])}")
            print(f"    Entities: {entities} | Rels: {rels}")

        elif mod == "pdf":
            print(f"  [{nid}] source={src} page={pg} | rels={rels}")
            print(f"    Content: {repr(content)}")
            print(f"    Entities: {entities}")

        elif mod == "image":
            ocr_status = meta.get("ocr_status", "?")
            ocr_text = meta.get("ocr_text", "")
            print(f"  [{nid}] source={src} | ocr_status={ocr_status} | ocr={repr(ocr_text[:60])}")
            print(f"    Entities: {entities}")

        print()

# Cross-video relationship check
print("=== CROSS-VIDEO RELATIONSHIP AUDIT ===")
problems = 0
id_to_src = {n["id"]: n.get("source","?") for n in ev}
for n in ev:
    src = n.get("source","?")
    for rel_id in n.get("relationships",[]):
        rel_src = id_to_src.get(rel_id, "MISSING")
        if rel_src == "MISSING":
            print(f"  BROKEN REL: {n['id']} -> {rel_id} (target not found)")
            problems += 1
        elif rel_src != src and n["modality"] in ("audio","video_frame") and id_to_src.get(rel_id, src) in ("audio","video_frame"):
            # Only flag audio<->frame cross-video links, PDF/image cross-modal is expected
            rel_mod = next((x["modality"] for x in ev if x["id"]==rel_id), "?")
            if rel_mod in ("audio","video_frame"):
                print(f"  CROSS-VIDEO: {n['id']} ({src}) -> {rel_id} ({rel_src})")
                problems += 1

if problems == 0:
    print("  All relationships are clean - no cross-video leakage detected.")
print()
print(f"AUDIT COMPLETE. Total problems found: {problems}")
