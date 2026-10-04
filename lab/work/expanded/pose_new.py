"""Run the MMA detector and RTMW-133 on the added 20s UFC clips E-H."""
from __future__ import annotations

import os
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from rtmlib import RTMPose
from ultralytics import YOLO

ROOT = Path.cwd()
CLIP_DIR = ROOT / "outputs/expanded/clips"
OUT = ROOT / "outputs/expanded/poses/E-H.json"
DET_PATH = ROOT / "work/models/mma-detector/best.pt"
POSE_PATH = Path(os.path.expanduser(
    "~/.cache/rtmlib/hub/checkpoints/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
))
IDS = "EFGH"
TARGET_HZ = 12.0
DETECTOR_IMGSZ = 960
DETECTOR_CONF = 0.25
THREADS = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently change detector device")
    torch.set_num_threads(THREADS)

    det = YOLO(str(DET_PATH))
    det.to("mps")
    # Validate actual model placement before processing clips.
    det_device = str(next(det.model.parameters()).device)
    if not det_device.startswith("mps"):
        raise RuntimeError(f"detector did not load on MPS: {det_device}")

    pose = RTMPose(
        str(POSE_PATH),
        model_input_size=(288, 384),
        backend="onnxruntime",
        device="cpu",
    )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    pose.session = ort.InferenceSession(
        str(POSE_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    providers = pose.session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"unexpected RTMW providers: {providers}")

    result = {
        "schema_version": 1,
        "method": "UFC person detector boxes followed by RTMW-133 per detected box; no training or identity labels",
        "sampling_hz": TARGET_HZ,
        "sampling": "Explicit native frame seeks at nearest frame to each k/12 second sample; no intervening native frames are decoded. Each selected frame is passed to inference at original 1280x720 pixels.",
        "detector": {
            "checkpoint": str(DET_PATH.relative_to(ROOT)),
            "checkpoint_sha256": sha256(DET_PATH),
            "imgsz": DETECTOR_IMGSZ,
            "confidence_threshold": DETECTOR_CONF,
            "device": det_device,
        },
        "pose_model": {
            "checkpoint": str(POSE_PATH),
            "checkpoint_sha256": sha256(POSE_PATH),
            "model_input_size": [288, 384],
            "keypoints": 133,
            "backend": "onnxruntime",
            "providers": providers,
            "device": "cpu",
            "intra_op_threads": THREADS,
            "inter_op_threads": 1,
            "score_units": "raw RTMW SIMCC peak score; not a calibrated probability",
        },
        "clips": {},
    }

    if OUT.exists():
        try:
            prior = json.loads(OUT.read_text())
            # Preserve completed clips from an interrupted batch only if the
            # inference settings still match this script.
            if (
                prior.get("sampling_hz") == TARGET_HZ
                and prior.get("detector", {}).get("checkpoint_sha256")
                == result["detector"]["checkpoint_sha256"]
                and prior.get("pose_model", {}).get("checkpoint_sha256")
                == result["pose_model"]["checkpoint_sha256"]
                and prior.get("detector", {}).get("imgsz") == DETECTOR_IMGSZ
            ):
                result["clips"].update(prior.get("clips", {}))
        except (OSError, json.JSONDecodeError):
            pass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for clip in result["clips"].values():
        clip.setdefault("inference_window_seconds", min(clip["duration_seconds"], 20.0))
    temp = OUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, separators=(",", ":")))
    temp.replace(OUT)
    for clip_id in IDS:
        if clip_id in result["clips"]:
            print(f"SKIP {clip_id}: already completed", flush=True)
            continue
        path = CLIP_DIR / f"{clip_id}.mp4"
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open clip {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = nframes / fps
        if width != 1280 or height != 720:
            raise RuntimeError(f"{clip_id}: expected 1280x720, got {width}x{height}")
        # Sampling targets lie in [0,duration); index choices are recorded exactly.
        window = min(duration, 20.0)
        target_times = np.arange(0.0, window, 1.0 / TARGET_HZ)
        sample_indices = np.rint(target_times * fps).astype(int)
        sample_indices = np.clip(sample_indices, 0, nframes - 1)
        sample_indices = np.unique(sample_indices)
        frames = []
        start = time.perf_counter()
        for seq, frame_idx in enumerate(sample_indices.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok or image.shape[:2] != (720, 1280):
                raise RuntimeError(f"{clip_id}: failed native-size read at frame {frame_idx}")
            prediction = det.predict(
                source=image,
                device="mps",
                imgsz=DETECTOR_IMGSZ,
                conf=DETECTOR_CONF,
                verbose=False,
            )[0]
            boxes = prediction.boxes.xyxy.detach().cpu().numpy()
            det_scores = prediction.boxes.conf.detach().cpu().numpy()
            if len(boxes):
                xy, pose_scores = pose(image, boxes.tolist())
            else:
                xy, pose_scores = [], []
            people = []
            for person_i, box in enumerate(boxes):
                kps = np.column_stack((xy[person_i], pose_scores[person_i]))
                people.append(
                    {
                        "box_xyxy_pixels": np.asarray(box).round(3).tolist(),
                        "detector_score": float(det_scores[person_i]),
                        "keypoints_xy_score": kps.round(3).tolist(),
                    }
                )
            frames.append(
                {
                    "sample_index": seq,
                    "frame_index": frame_idx,
                    "target_time_seconds": round(float(target_times[seq]), 9),
                    "capture_time_seconds": round(frame_idx / fps, 9),
                    "people": people,
                }
            )
            if seq % 24 == 0:
                print(f"{clip_id}: {seq}/{len(sample_indices)} samples", flush=True)
        cap.release()
        elapsed = time.perf_counter() - start
        result["clips"][clip_id] = {
            "source_path": str(path.relative_to(ROOT)),
            "source_sha256": sha256(path),
            "width": width,
            "height": height,
            "source_fps": fps,
            "native_frame_count": nframes,
            "duration_seconds": duration,
            "inference_window_seconds": window,
            "sample_count": len(frames),
            "inference_seconds": elapsed,
            "frames": frames,
        }
        temp = OUT.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, separators=(",", ":")))
        temp.replace(OUT)
        print(f"DONE {clip_id}: {len(frames)} samples in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
