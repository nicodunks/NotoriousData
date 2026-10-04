#!/usr/bin/env python3
"""Run the released fight gate and phase model on manifest-listed videos.

Each video is split into five-second windows, matching the released analyzer.
The script records every gate probability and, for windows retained by the gate,
the full phase probability vector. It intentionally omits detector/identity work.

Manifest formats:
  * JSON list of {"id": "...", "path": "...", ...} objects
  * JSON object with a "clips" list
  * CSV with at least a "path" column; optional "id" and annotation columns
Paths are resolved relative to --clips-dir unless absolute.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

REPO = Path(__file__).resolve().parent / "mma-fight-analyzer"
sys.path.insert(0, str(REPO / "src"))
from mma import config as C  # noqa: E402
from mma import identity as ident  # noqa: E402
from mma.models import MODEL_INPUT_STATS, load_gate_model, load_phase_model  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            records = list(csv.DictReader(f))
    else:
        obj = json.loads(path.read_text())
        records = obj.get("clips", []) if isinstance(obj, dict) else obj
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a list of clip records")
    for i, record in enumerate(records):
        if not isinstance(record, dict) or not (record.get("path") or record.get("file")):
            raise ValueError(f"Manifest record {i} needs a path field")
        record.setdefault("id", Path(record.get("path") or record["file"]).stem)
    return records


def frame_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().div_(255).permute(2, 0, 1)


@torch.inference_mode()
def gate_probability(model, frames: list[np.ndarray], device: torch.device) -> float:
    idx = np.linspace(0, len(frames) - 1, C.GATE_FRAMES).round().astype(int)
    mean, std = C.IMAGENET_MEAN, C.IMAGENET_STD
    batch = []
    for i in idx:
        t = TF.resize(frame_tensor(frames[i]), [C.CROP_SIZE, C.CROP_SIZE], antialias=True)
        batch.append(TF.normalize(t, mean, std))
    return torch.sigmoid(model(torch.stack(batch).to(device))).mean().item()


@torch.inference_mode()
def phase_probabilities(model, meta: dict, frames: list[np.ndarray], mask: np.ndarray, device: torch.device):
    mean, std = MODEL_INPUT_STATS[meta["model_name"]]
    rgb = torch.stack([frame_tensor(f) for f in frames])
    rgb = TF.resize(rgb, [C.CROP_SIZE, C.CROP_SIZE], antialias=True)
    rgb = TF.normalize(rgb, mean, std)
    video = rgb.permute(1, 0, 2, 3)
    if meta["in_channels"] == 4:
        m = torch.from_numpy(mask).float().unsqueeze(1)
        m = TF.resize(m, [C.CROP_SIZE, C.CROP_SIZE], interpolation=TF.InterpolationMode.NEAREST)
        video = torch.cat((video, m.permute(1, 0, 2, 3)), dim=0)
    logits, _ = model(video.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return {label: float(prob) for label, prob in zip(C.PHASE_LABELS, probs)}


def pose_mask(pose_frames: list[dict], source_frame_indices: list[int], frame_hw: tuple[int, int]):
    """Build the trained +/- fighter mask from pose person boxes.

    The pose detector supplies boxes but no fighter identity. Within each window
    we link boxes by IoU and assign the initial left-to-right pair consistently;
    the phase model does not use pressure labels. Overlap and missing detections
    remain zero, matching the released mask rasterizer.
    """
    dets = []
    for fi in source_frame_indices:
        people = pose_frames[fi].get("people", []) if 0 <= fi < len(pose_frames) else []
        boxes = [tuple(map(int, p["box"])) for p in people if p.get("box") and len(p["box"]) == 4]
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        dets.append(boxes[:2])
    tracks = ident._link_tracks(dets)
    tracks = sorted(tracks, key=lambda tr: sum((b[2]-b[0])*(b[3]-b[1]) for b in tr.values()), reverse=True)[:2]
    tracks.sort(key=lambda tr: tr[min(tr)][0] if tr else 0)
    assignments = []
    for ti in range(C.NUM_FRAMES):
        pair = []
        for tr in tracks:
            pair.append(tr.get(ti))
        assignments.append((pair[0] if pair else None, pair[1] if len(pair) > 1 else None))
    h, w = frame_hw
    mh, mw = C.CACHE_SHORT_SIDE, round(C.CACHE_SHORT_SIDE * w / h)
    mask = ident.build_masks((h, w), assignments, (mh, mw))
    return mask.astype(np.float32), len(tracks)


def read_windows(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_per_window = max(1, int(round(fps * C.CLIP_SECONDS)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    windows = []
    current = []
    while True:
        ok, frame = cap.read()
        if ok:
            current.append(frame)
        if (not ok and current) or len(current) == frames_per_window:
            windows.append(current)
            current = []
        if not ok:
            break
    cap.release()
    return windows, fps, frame_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clips-dir", type=Path, default=Path("work/clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phases"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    gate_path = REPO / "outputs/gate/gate.pt"
    phase_path = REPO / "outputs/phase/deployment_phase_final.pt"
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    gate, gate_meta = load_gate_model(gate_path, device)
    phase, phase_meta = load_phase_model(phase_path, device)
    gate_threshold = float(gate_meta.get("threshold", 0.5))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.manifest)
    print(f"Device: {device}; gate threshold: {gate_threshold}; phase input: {phase_meta}", flush=True)

    for record in records:
        value = record.get("path") or record.get("file")
        video_path = Path(value)
        if not video_path.is_absolute():
            video_path = args.clips_dir / video_path
        started = time.perf_counter()
        windows, fps, frame_count = read_windows(video_path)
        pose_path = Path("outputs/pose") / f"{record['id']}__mma.json"
        pose_data = json.loads(pose_path.read_text()) if pose_path.exists() else None
        pose_frames = pose_data.get("frames", []) if pose_data else []
        results = []
        for wi, frames in enumerate(windows):
            sample_idx = np.linspace(0, len(frames) - 1, C.NUM_FRAMES).round().astype(int)
            sampled = [frames[i] for i in sample_idx]
            p_excluded = gate_probability(gate, sampled, device)
            passed = p_excluded <= gate_threshold
            item = {
                "window": wi,
                "start_s": round(wi * C.CLIP_SECONDS, 3),
                "duration_s": round(len(frames) / fps, 3),
                "gate_prob_excluded": p_excluded,
                "gate_threshold": gate_threshold,
                "gate_pass": passed,
            }
            if passed:
                global_indices = [wi * max(1, int(round(fps * C.CLIP_SECONDS))) + int(i) for i in sample_idx]
                mask, track_count = pose_mask(pose_frames, global_indices, frames[0].shape[:2]) if pose_frames else (np.zeros((C.NUM_FRAMES, C.CACHE_SHORT_SIDE, round(C.CACHE_SHORT_SIDE * frames[0].shape[1] / frames[0].shape[0])), np.float32), 0)
                probs = phase_probabilities(phase, phase_meta, sampled, mask, device)
                item["phase_probabilities"] = probs
                item["phase_prediction"] = max(probs, key=probs.get)
                swapped = phase_probabilities(phase, phase_meta, sampled, -mask, device)
                zero = phase_probabilities(phase, phase_meta, sampled, np.zeros_like(mask), device)
                item["phase_mask_ablations"] = {
                    "swapped_signs": {"probabilities": swapped, "prediction": max(swapped, key=swapped.get)},
                    "zero_mask": {"probabilities": zero, "prediction": max(zero, key=zero.get)},
                }
                item["mask_source"] = "MMA pose boxes, IoU-linked left-to-right assignment" if pose_frames else "zero mask (pose detections unavailable)"
                item["mask_tracks"] = track_count
            results.append(item)
            print(f"{record['id']} [{item['start_s']:g}s]: gate_excluded={p_excluded:.4f}, "
                  f"phase={item.get('phase_prediction', 'skipped')}", flush=True)
        payload = {
            "clip_id": str(record["id"]),
            "source_path": str(video_path),
            "source_metadata": {k: v for k, v in record.items() if k not in ("path", "file", "id")},
            "fps": fps,
            "source_frame_count": frame_count,
            "window_seconds": C.CLIP_SECONDS,
            "frame_sampling": {"frames_per_window": C.NUM_FRAMES, "gate_frames": C.GATE_FRAMES},
            "gate_checkpoint": {
                "path": str(gate_path), "sha256": sha256(gate_path), "meta": gate_meta,
            },
            "phase_checkpoint": {
                "path": str(phase_path), "sha256": sha256(phase_path), "meta": phase_meta,
            },
            "pose_checkpoint": ({"path": pose_data.get("checkpoint"), "sha256": pose_data.get("checkpoint_sha256"), "model": pose_data.get("model")} if pose_data else None),
            "identity_mask_note": "MMA pose boxes are IoU-linked; the two largest tracks are assigned +1/-1 left-to-right at each window start. This does not establish broadcast Fighter 1/Fighter 2 identity; swapped-sign and zero-mask phase ablations are included.",
            "device": str(device),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "windows": results,
        }
        out = args.output_dir / f"{record['id']}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
