"""
Football Ball Tracker — RF-DETR Nano (Local) + ByteTrack
=========================================================
Requirements:
    pip install trackers[detection] supervision rfdetr opencv-python

Usage:
    # Webcam (default)
    python ball_tracker.py

    # Video file
    python ball_tracker.py --source football_match.mp4

    # Save output
    python ball_tracker.py --source football_match.mp4 --output tracked_output.mp4

    # Filter specific class (e.g., class 0 = ball in your model)
    python ball_tracker.py --source football_match.mp4 --class-filter 0

    # Adjust detection confidence
    python ball_tracker.py --source football_match.mp4 --confidence 0.2
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase         # RF-DETR local model import
from trackers import ByteTrackTracker


# ─────────────────────────────────────────────
#  Argument Parsing
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR Nano + ByteTrack Live Tracker")

    parser.add_argument(
        "--source",
        type=str,
        default="0",                        # 0 = webcam
        help="Video source: webcam index (0), video file path, or RTSP URL",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=None,                       # None = use RF-DETR default nano weights
        help="Path to your local RF-DETR nano .pth weights file",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (lower = more detections, 0.15–0.4 recommended)",
    )
    parser.add_argument(
        "--class-filter",
        type=int,
        default=37,           # 37 = "sports ball" in COCO — RF-DETR nano default class ID for ball
        help="Filter detections to a single class ID. Default=37 (ball in RF-DETR/COCO). Pass -1 to disable filtering.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (e.g., output.mp4). If not set, video is only displayed.",
    )
    parser.add_argument(
        "--lost-track-buffer",
        type=int,
        default=30,
        help="Frames to keep a lost track alive (higher = survives longer occlusions)",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=1,
        help="Min frames before a track is confirmed (1 = immediate, good for fast ball)",
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        default=True,
        help="Draw trajectory trail behind tracked objects",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Suppress live display window (useful for headless servers)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Model Loader
# ─────────────────────────────────────────────
def load_model(weights_path: str | None):
    """Load RF-DETR nano from local weights or default pretrained."""
    print("[INFO] Loading RF-DETR Nano model...")

    if weights_path:
        path = Path(weights_path)
        if not path.exists():
            print(f"[ERROR] Weights file not found: {weights_path}")
            sys.exit(1)
        # Load with custom weights path
        model = RFDETRBase(pretrain_weights=str(path))
        print(f"[INFO] Loaded local weights: {weights_path}")
    else:
        # Uses default RF-DETR nano pretrained weights
        model = RFDETRBase()
        print("[INFO] Loaded default RF-DETR pretrained weights")

    return model


# ─────────────────────────────────────────────
#  Video Source
# ─────────────────────────────────────────────
def open_video_source(source: str) -> cv2.VideoCapture:
    """Open webcam, video file, or RTSP stream."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    source_type = "webcam" if isinstance(src, int) else "video/stream"
    print(f"[INFO] Opened {source_type}: {source}")
    return cap


# ─────────────────────────────────────────────
#  Video Writer Setup
# ─────────────────────────────────────────────
def setup_writer(output_path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter | None:
    if not output_path:
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[INFO] Saving output to: {output_path} ({width}x{height} @ {fps:.1f} fps)")
    return writer


# ─────────────────────────────────────────────
#  Annotators
# ─────────────────────────────────────────────
def setup_annotators(show_trajectory: bool):
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.6,
        text_thickness=1,
        text_padding=4,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=40,           # trail length in frames
    ) if show_trajectory else None

    return box_annotator, label_annotator, trace_annotator


# ─────────────────────────────────────────────
#  FPS Counter
# ─────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────
#  Draw HUD (stats overlay)
# ─────────────────────────────────────────────
def draw_hud(frame: np.ndarray, fps: float, num_tracks: int) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (240, 65), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(frame, f"FPS:    {fps:5.1f}",       (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),   1, cv2.LINE_AA)
    cv2.putText(frame, f"Tracks: {num_tracks}",     (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # --- Setup ---
    model   = load_model(args.model_weights)
    tracker = ByteTrackTracker(
        lost_track_buffer=args.lost_track_buffer,
        minimum_consecutive_frames=args.min_consecutive_frames,
    )

    cap    = open_video_source(args.source)
    writer = setup_writer(args.output, cap)

    box_annotator, label_annotator, trace_annotator = setup_annotators(args.show_trajectory)
    fps_counter = FPSCounter()

    print("\n[INFO] Starting tracking. Press 'Q' to quit, 'S' to save a screenshot.\n")
    screenshot_count = 0

    # --- Frame Loop ---
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        # RF-DETR expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Detection ──────────────────────────────
        detections = model.predict(
            frame_rgb,
            threshold=args.confidence
        )

        # ── Optional Class Filter ──────────────────
        if args.class_filter is not None and args.class_filter >= 0 and len(detections) > 0:
            mask = detections.class_id == args.class_filter
            detections = detections[mask]

        # ── Tracking ───────────────────────────────
        detections = tracker.update(detections)

        # ── Build Labels ───────────────────────────
        labels = []
        for i in range(len(detections)):
            tid   = detections.tracker_id[i] if detections.tracker_id is not None else "?"
            conf  = detections.confidence[i] if detections.confidence is not None else 0.0
            label = f"#{tid}  {conf:.2f}"
            labels.append(label)

        # ── Annotate Frame ─────────────────────────
        annotated = frame_bgr.copy()

        if trace_annotator and len(detections) > 0:
            annotated = trace_annotator.annotate(annotated, detections)

        if len(detections) > 0:
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

        # ── HUD Overlay ────────────────────────────
        fps = fps_counter.tick()
        annotated = draw_hud(annotated, fps, len(detections))

        # ── Save Frame ────────────────────────────
        if writer:
            writer.write(annotated)

        # ── Display ────────────────────────────────
        if not args.no_display:
            cv2.imshow("RF-DETR Nano + ByteTrack | Press Q to quit", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:     # Q or ESC
                print("[INFO] Quit signal received.")
                break
            elif key == ord("s"):                # S = screenshot
                screenshot_count += 1
                fname = f"screenshot_{screenshot_count:04d}.jpg"
                cv2.imwrite(fname, annotated)
                print(f"[INFO] Screenshot saved: {fname}")

    # --- Cleanup ---
    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Output saved: {args.output}")
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
