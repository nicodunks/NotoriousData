#!/usr/bin/env python3
"""Run BloodshotNet on the locally prepared sample clips and save raw detections."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
CLIPS = WORK / "clips"
OUT = ROOT / "outputs" / "blood"
REPO = "dennis-at-bit/BloodshotNet"
REVISION = "dc8d0885daae3b64c5da7f691f7d9591ac49c43f"
WEIGHT = "yolo26n.pt"
CONF = 0.25
SAMPLE_FPS = 3


def entries_from_manifest(data):
    if isinstance(data, list):
        return data
    for key in ("samples", "clips", "items", "videos"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    return []


def get_clip_path(entry):
    value = None
    if isinstance(entry, str):
        value = entry
    elif isinstance(entry, dict):
        for key in ("clip", "path", "file", "video", "filename"):
            if entry.get(key):
                value = entry[key]
                break
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        # Manifest paths are usually relative to the workspace or work directory.
        candidates = (ROOT / p, WORK / p, CLIPS / p, CLIPS / p.name)
        p = next((c for c in candidates if c.exists()), candidates[0])
    return p


def main():
    manifest_path = WORK / "samples.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing sample manifest: {manifest_path}")
    entries = entries_from_manifest(json.loads(manifest_path.read_text()))
    paths = []
    for entry in entries:
        p = get_clip_path(entry)
        if p and p.exists() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            paths.append((entry, p))
    if not paths:
        paths = [(None, p) for p in sorted(CLIPS.glob("*.mp4"))]
    if not paths:
        raise SystemExit(f"No sample clips found under {CLIPS}")

    OUT.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=REPO,
        filename=WEIGHT,
        revision=REVISION,
        cache_dir=str(WORK / "hf-cache"),
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(model_path)
    try:
        class_names = {str(k): v for k, v in model.names.items()}
    except Exception:
        class_names = {}

    run_started = time.perf_counter()
    for entry, clip_path in paths:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Skipping unreadable clip: {clip_path}")
            continue
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / source_fps if source_fps > 0 else None
        step = max(1, round(source_fps / SAMPLE_FPS)) if source_fps > 0 else 1
        detections = []
        frame_timings = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                local_time = frame_index / source_fps if source_fps > 0 else None
                if device == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                result = model.predict(
                    source=frame,
                    conf=CONF,
                    device=device,
                    verbose=False,
                )[0]
                if device == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - t0
                frame_timings.append({"local_time_seconds": local_time, "inference_seconds": elapsed})
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = [round(float(x), 3) for x in box.xyxy[0].tolist()]
                        cls_id = int(box.cls[0].item())
                        detections.append({
                            "local_time_seconds": local_time,
                            "frame_index": frame_index,
                            "bbox_xyxy": xyxy,
                            "confidence": round(float(box.conf[0].item()), 6),
                            "class_id": cls_id,
                            "class_name": class_names.get(str(cls_id), str(cls_id)),
                        })
            frame_index += 1
        cap.release()
        stem = clip_path.stem
        clip_elapsed = sum(x["inference_seconds"] for x in frame_timings)
        payload = {
            "clip": str(clip_path.relative_to(ROOT)) if clip_path.is_relative_to(ROOT) else str(clip_path),
            "sample": entry,
            "source_fps": source_fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "sample_fps_requested": SAMPLE_FPS,
            "sample_stride_frames": step,
            "confidence_threshold": CONF,
            "device": device,
            "model": {"repo_id": REPO, "filename": WEIGHT, "revision": REVISION, "path": model_path},
            "class_names": class_names,
            "sampled_frames": len(frame_timings),
            "inference_seconds_total": round(clip_elapsed, 6),
            "inference_seconds_mean": round(clip_elapsed / len(frame_timings), 6) if frame_timings else None,
            "detections": detections,
            "frame_timings": frame_timings,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        out_path = OUT / f"{stem}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{clip_path.name}: {len(frame_timings)} frames, {len(detections)} detections, {clip_elapsed:.2f}s -> {out_path}")
    print(f"All clips elapsed: {time.perf_counter() - run_started:.2f}s; device={device}")


if __name__ == "__main__":
    main()
