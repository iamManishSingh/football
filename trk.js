
def _minmax_norm(x):
  x = np.asarray(x, dtype=np.float32)
  if x.size < 2:
    return x
  min_x = np.min(x)
  max_x = np.max(x)
  if max_x == min_x:
    return np.ones_like(x)
  return (x - min_x) / (max_x - min_x)

# fr_detr_strongsort_loader.py
# RF-DETR loader + StrongSORT tracking (BoxMOT).
# - Guarantees predict_batch(images, threshold=...) exists
# - Adds per-frame StrongSORT tracking
# - Returns track IDs as "keypoints" in a safe iterable shape: (N, 1, 3) -> [track_id, confirmed_flag, 1]
# - Preserves any original RF-DETR keypoints as "pose_keypoints"
# - Includes predicted (unmatched) tracker boxes directly in xyxy/confidence/class_id + keypoints:
#     confirmed_flag = 1 for detection-confirmed boxes, 0 for predicted-only boxes

import re
import os
import numpy as np
import builtins

from rfdetr import (
  RFDETRBase,
  RFDETRLarge,
  RFDETRNano,
  RFDETRSmall,
  RFDETRMedium,
  RFDETRSegPreview,
)
from rfdetr.util.coco_classes import COCO_CLASSES  # dict[id] -> str


_real_print = builtins.print

def _filtered_print(*args, **kwargs):
  msg = " ".join(str(a) for a in args)
  if "not a problem" in msg or "pretrain weights" in msg:
    return
  _real_print(*args, **kwargs)

builtins.print = _filtered_print


def _to_snake(name):
  s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
  return re.sub(r"_+", "_", s).strip("_")


SIMPLIFIED_CLASS_NAMES = {i: _to_snake(n) for i, n in COCO_CLASSES.items()}

_NAME_TO_CTOR = {
  "base": RFDETRBase, "large": RFDETRLarge, "nano": RFDETRNano,
  "small": RFDETRSmall, "medium": RFDETRMedium,
  "segpreview": RFDETRSegPreview, "seg_preview": RFDETRSegPreview, "seg": RFDETRSegPreview,
}
_CLASSNAME_TO_CTOR = {
  "RFDETRBase": RFDETRBase, "RFDETRLarge": RFDETRLarge, "RFDETRNano": RFDETRNano,
  "RFDETRSmall": RFDETRSmall, "RFDETRMedium": RFDETRMedium, "RFDETRSegPreview": RFDETRSegPreview,
}


def _ensure_uint8_hwc(img):
  arr = np.asarray(img)
  if arr.ndim != 3 or arr.shape[2] != 3:
    raise ValueError(f"Expected HWC image with 3 channels, got shape={arr.shape}")
  if arr.dtype == np.uint8:
    return arr
  if np.issubdtype(arr.dtype, np.floating):
    mx = float(np.nanmax(arr)) if arr.size else 0.0
    if mx <= 1.5:
      arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)
  return np.clip(arr, 0, 255).astype(np.uint8)


def _normalize_rfdetr_item(det, threshold, base_confidence=0.2):
  if isinstance(det, dict):
    xyxy = det.get("xyxy"); conf = det.get("confidence"); cls = det.get("class_id")
    kps  = det.get("keypoints", None)
  else:
    xyxy = getattr(det, "xyxy", None); conf = getattr(det, "confidence", None)
    cls  = getattr(det, "class_id", None); kps = getattr(det, "keypoints", None)

  xyxy = np.asarray(xyxy if xyxy is not None else [], dtype=np.float32).reshape(-1, 4)
  conf = np.asarray(conf if conf is not None else [], dtype=np.float32).reshape(-1)
  cls  = np.asarray(cls  if cls  is not None else [], dtype=np.int64).reshape(-1)

  if kps is not None:
    kps = np.asarray(kps, dtype=np.float32)

  if conf.size:
    mask = conf >= float(threshold)
    xyxy, conf, cls = xyxy[mask], conf[mask], cls[mask]
    if kps is not None:
      kps = kps[mask]
    conf_norm = _minmax_norm(conf)
    min_conf = np.average(conf_norm) if conf_norm.size else 0.0
    shift_by = 0.75 - min_conf
    conf = conf_norm + (shift_by if shift_by > 0 else 0.0)


  out = {"xyxy": xyxy, "confidence": conf, "class_id": cls}
  if kps is not None:
    out["keypoints"] = kps
  return out


def _coerce_tracks_to_2d(tracks):
  """
  Make tracker.update(...) output safe to iterate.

  Canonical expected: (N,8)
    [x1,y1,x2,y2,id,conf,cls,det_ind]

  Accepts None, scalar, 1x8, Nx8; unwraps (tracks, ...) tuples.
  """
  if tracks is None:
    return np.zeros((0, 8), dtype=np.float32)

  if isinstance(tracks, (tuple, list)) and tracks:
    tracks = tracks[0]

  arr = np.asarray(tracks)

  if arr.ndim == 0:
    return np.zeros((0, 8), dtype=np.float32)

  if arr.ndim == 1:
    if arr.size == 0:
      return np.zeros((0, 8), dtype=np.float32)
    if arr.size >= 8:
      return arr.astype(np.float32, copy=False).reshape(1, -1)[:, :8]
    return np.zeros((0, 8), dtype=np.float32)

  arr2 = arr.reshape(-1, arr.shape[-1])
  if arr2.shape[1] < 8:
    return np.zeros((0, 8), dtype=np.float32)
  return arr2[:, :8].astype(np.float32, copy=False)


def _make_track_keypoints(track_ids, confirmed_flags=None):
  """
  Pose-like format: (N, 1, 3) with [x, y, score]
    x = track_id
    y = confirmed flag (1 = detection confirmed, 0 = predicted-only)
    score = 1
  """
  track_id = np.asarray(track_ids, dtype=np.float32).reshape(-1)

  if confirmed_flags is None:
    confirmed = np.zeros_like(track_id, dtype=np.float32)
  else:
    confirmed = np.asarray(confirmed_flags, dtype=np.float32).reshape(-1)
    if confirmed.shape[0] != track_id.shape[0]:
      # Safe fallback
      confirmed = np.zeros_like(track_id, dtype=np.float32)

  score = np.ones_like(track_id, dtype=np.float32)
  kps = np.stack([track_id, confirmed, score], axis=1)  # (N,3)
  return kps.reshape(-1, 1, 3)  # (N,1,3)


def _resolve_tracker_device(device, tracker_device):
  """
  BoxMOT wants:
    - 'cpu' OR
    - CUDA device indices as string: '0' or '0,1,...'
  It does NOT accept 'cuda' if no devices exist.

  If user has CUDA build but no GPUs (device_count==0), force cpu.
  """
  if tracker_device is not None:
    td = str(tracker_device).strip()
  else:
    td = str(device).strip() if (device and device != "auto") else ""

  # Normalize common forms
  if td.startswith("cuda:"):
    td = td.split("cuda:", 1)[1].strip()  # '0'
  if td == "cuda":
    td = "0"  # only valid if a GPU exists

  # Detect real GPU availability
  has_gpu = False
  try:
    import torch
    has_gpu = bool(torch.cuda.is_available())
  except Exception:
    has_gpu = False

  if not has_gpu:
    return "cpu"

  if td == "" or td == "auto":
    return "0"

  # If already looks like '0' or '0,1'
  if all(part.isdigit() for part in td.split(",") if part != ""):
    return td

  # Fallback
  return "0"


class _RFDetrBatchWrapper:
  def __init__(self, core):
    self._core = core
    setattr(self, "_runner_label_map", dict(SIMPLIFIED_CLASS_NAMES))

  def __getattr__(self, name):
    return getattr(self._core, name)

  def predict_batch(self, images, threshold=0.1):
    core = self._core

    pred = getattr(core, "predict", None)
    if callable(pred):
      return pred(images, threshold=threshold)

    pred_batch = getattr(core, "predict_batch", None)
    if callable(pred_batch):
      return pred_batch(images, threshold=threshold)

    try:
      return core(images, threshold=threshold)  # type: ignore
    except Exception:
      pass

    if callable(pred):
      return [pred(img, threshold=threshold) for img in images]

    try:
      return [core(img, threshold=threshold) for img in images]  # type: ignore
    except Exception as e:
      raise TypeError(f"RF-DETR core lacks a usable prediction method: {e}")


class _StrongSortAdapter:
  def __init__(
    self,
    reid_weights,
    device="auto",
    half=False,
    per_class=False,
    tracker_config_overrides=None
  ):
    try:
      from boxmot import StrongSort, DeepOcSort, BoostTrack, BotSort, HybridSort, OcSort, ByteTrack
    except Exception as e:
      raise ImportError("StrongSORT requested but BoxMOT is not available. Install with: pip install boxmot") from e

    if not reid_weights:
      raise ValueError(
        "StrongSORT requires reid_weights (path to a ReID model). "
        "Pass reid_weights=... to create_model(...) or set STRONGSORT_REID_WEIGHTS."
      )

    # cfg = get_tracker_config("deepocsort")

    # cfg["det_thresh"] = 0.3
    # cfg["iou_threshold"] = 0.7
    # cfg["min_hits"] = 2

    # if isinstance(tracker_config_overrides, dict):
    #   cfg.update(tracker_config_overrides)

    self._tracker = BotSort(
      reid_weights=reid_weights,
      device=device,
      half=half,
      with_longterm_reid_correction=True,
      longterm_reid_correction_thresh=0.25,
      longterm_reid_correction_thresh_low=0.20,

      EG_weight_high_score=2.0,     # down from 4.6 :contentReference[oaicite:15]{index=15}
      EG_weight_low_score=0,      # key change (prevents low-score ping-pong) :contentReference[oaicite:16]{index=16}

      iou_threshold=0.1,           # up from 0.3 :contentReference[oaicite:17]{index=17}
      alpha=0.97,
      adapfs=True,
      max_age=30*60,
    )

  def update(self, dets_mx6, img_rgb_uint8):
    dets = np.asarray(dets_mx6, dtype=np.float32).reshape(-1, 6)
    img = _ensure_uint8_hwc(img_rgb_uint8)
    return self._tracker.update(dets, img)


class _RFDetrStrongSortWrapper:
  def __init__(self, det_model, tracker, images_are_bgr=True, track_ids_as_keypoints=True):
    self._det_model = det_model
    self._tracker = tracker
    self._images_are_bgr = bool(images_are_bgr)
    self._track_ids_as_keypoints = bool(track_ids_as_keypoints)

    if hasattr(det_model, "_runner_label_map"):
      setattr(self, "_runner_label_map", getattr(det_model, "_runner_label_map"))

  def __getattr__(self, name):
    return getattr(self._det_model, name)

  def predict_batch(self, images, threshold=0.1):
    raw = self._det_model.predict_batch(images, threshold=threshold)
    if not isinstance(raw, (list, tuple)):
      raw = [raw]

    out = []

    for img, det in zip(images, raw):
      norm = _normalize_rfdetr_item(det, threshold=threshold)

      det_xyxy = norm["xyxy"]
      det_conf = norm["confidence"]
      det_cls  = norm["class_id"]

      if det_xyxy.size:
        dets = np.concatenate(
          [
            det_xyxy.astype(np.float32),
            det_conf.reshape(-1, 1),
            det_cls.astype(np.float32).reshape(-1, 1),
          ],
          axis=1
        )
      else:
        dets = np.zeros((0, 6), dtype=np.float32)

      img_uint8 = _ensure_uint8_hwc(img)
      img_rgb = img_uint8[:, :, ::-1] if self._images_are_bgr else img_uint8

      tracks_raw = self._tracker.update(dets, img_rgb)
      tracks = _coerce_tracks_to_2d(tracks_raw)

      det_count = int(dets.shape[0])

      # For the detection rows (0..det_count-1), assign track IDs when available.
      det_track_id = np.full((det_count,), -1, dtype=np.int64)
      det_confirmed = np.ones((det_count,), dtype=np.float32)  # detection-confirmed

      # Predicted-only tracks (unmatched to a detection this frame)
      pred_xyxy = np.zeros((0, 4), dtype=np.float32)
      pred_track_id = np.zeros((0,), dtype=np.int64)
      pred_conf = np.zeros((0,), dtype=np.float32)
      pred_cls = np.zeros((0,), dtype=np.int64)
      pred_confirmed = np.zeros((0,), dtype=np.float32)  # predicted-only

      if tracks.shape[0]:
        det_ind = tracks[:, 7].astype(np.int64, copy=False)
        matched_mask = (det_ind >= 0) & (det_ind < det_count)
        predicted_mask = ~matched_mask

        if np.any(matched_mask):
          matched_rows = tracks[matched_mask]
          for row in matched_rows:
            det_index = int(row[7])
            if 0 <= det_index < det_track_id.shape[0]:
              det_track_id[det_index] = int(row[4])
              # Use tracker-refined box for that detection
              det_xyxy[det_index] = row[0:4].astype(np.float32)

        if np.any(predicted_mask):
          pred_rows = tracks[predicted_mask]
          pred_xyxy = pred_rows[:, 0:4].astype(np.float32, copy=False)
          pred_track_id = pred_rows[:, 4].astype(np.int64, copy=False)
          pred_conf = pred_rows[:, 5].astype(np.float32, copy=False)
          pred_cls = pred_rows[:, 6].astype(np.int64, copy=False)
          pred_confirmed = np.zeros((pred_rows.shape[0],), dtype=np.float32)

      # Preserve original pose keypoints (if any) and pad for predicted rows
      if self._track_ids_as_keypoints and "keypoints" in norm:
        pose_kps = np.asarray(norm["keypoints"], dtype=np.float32)
        if pose_kps.ndim >= 1 and pose_kps.shape[0] == det_count and pred_xyxy.shape[0]:
          pad_shape = (int(pred_xyxy.shape[0]),) + tuple(pose_kps.shape[1:])
          pad = np.full(pad_shape, np.nan, dtype=np.float32)
          norm["pose_keypoints"] = np.concatenate([pose_kps, pad], axis=0)
        else:
          norm["pose_keypoints"] = pose_kps

      # Merge detections + predicted into the standard outputs (no extra fields)
      if pred_xyxy.shape[0]:
        norm["xyxy"] = np.concatenate([det_xyxy, pred_xyxy], axis=0)
        norm["confidence"] = np.concatenate([det_conf.astype(np.float32, copy=False), pred_conf], axis=0)
        norm["class_id"] = np.concatenate([det_cls.astype(np.int64, copy=False), pred_cls], axis=0)
        merged_track_id = np.concatenate([det_track_id, pred_track_id], axis=0)
        merged_confirmed = np.concatenate([det_confirmed, pred_confirmed], axis=0)
      else:
        norm["xyxy"] = det_xyxy
        norm["confidence"] = det_conf
        norm["class_id"] = det_cls
        merged_track_id = det_track_id
        merged_confirmed = det_confirmed

      if self._track_ids_as_keypoints:
        norm["keypoints"] = _make_track_keypoints(merged_track_id, confirmed_flags=merged_confirmed)
      else:
        norm["track_id"] = merged_track_id

      out.append(norm)

    return out


def create_model(
  model_spec,
  device=None,
  optimize=True,
  enable_tracking=True,
  reid_weights="clip_market1501.pt",
  tracker_device=None,
  half=None,
  per_class=True,
  tracker_config_overrides=None,
  images_are_bgr=True,
  track_ids_as_keypoints=True,
):
  key = str(model_spec).strip()
  ctor = _NAME_TO_CTOR.get(key.lower()) or _CLASSNAME_TO_CTOR.get(key) or RFDETRSegPreview
  core = ctor()

  # Move detector if explicitly requested; if user passes 'cuda' but no devices exist, this may fail.
  if hasattr(core, "to") and device and device != "auto":
    try:
      core = core.to(device)
    except Exception:
      pass

  if optimize and hasattr(core, "optimize_for_inference"):
    try:
      core.optimize_for_inference(compile=False)
    except Exception:
      pass

  det_model = _RFDetrBatchWrapper(core)

  if not enable_tracking:
    return det_model

  if reid_weights is None:
    reid_weights = os.environ.get("STRONGSORT_REID_WEIGHTS", None)

  resolved_tracker_device = _resolve_tracker_device(device, tracker_device)

  if half is None:
    half = resolved_tracker_device != "cpu"

  tracker = _StrongSortAdapter(
    reid_weights=reid_weights,
    device=resolved_tracker_device,
    half=bool(half),
    per_class=bool(per_class),
    tracker_config_overrides=tracker_config_overrides,
  )

  return _RFDetrStrongSortWrapper(
    det_model=det_model,
    tracker=tracker,
    images_are_bgr=images_are_bgr,
    track_ids_as_keypoints=track_ids_as_keypoints,
  )


# force runner’s batch path; no tensor preprocess needed
preprocess = None


def adapter(raw, imgs, threshold):
  if not isinstance(raw, (list, tuple)):
    raw = [raw]
  out = []
  for det in raw:
    out.append(_normalize_rfdetr_item(det, threshold=threshold))
  return out
