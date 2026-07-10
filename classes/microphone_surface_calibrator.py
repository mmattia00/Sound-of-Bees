#!/usr/bin/env python3
"""
Microphone surface calibration helper for the two-camera setup.

The calibrator associates microphone IDs with pixel coordinates on the cam-0
and cam-1 frames, saves the mapping to JSON, and later exposes lookup helpers
used by the synchronization pipeline and the debug highlight workflow.

Two-camera calibration:
  - cam-1 covers mic IDs  1..16  (front face)
  - cam-0 covers mic IDs 17..32  (back face)

Usage (calibrate):
    python microphone_surface_calibrator.py calibrate
        --video-folder PATH
        --output PATH          [default: calibration.json]
        --num-mics-per-cam 16

Usage (highlight test):
    python microphone_surface_calibrator.py highlight
        --calibration PATH
        --mic-id INT
        --video-folder PATH    (grabs a frame to draw on)
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CamCalibration:
    cam: int                                    # 0 or 1
    mic_ids: list[int]                          # which mic IDs belong to this cam
    microphone_points: dict[int, tuple[int, int]] = field(default_factory=dict)  # mic_id -> (x, y) in cam frame


@dataclass
class CalibrationData:
    num_mics: int
    cam0: CamCalibration   # mic IDs 17..32
    cam1: CamCalibration   # mic IDs  1..16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_gui_display() -> bool:
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _pick_random_video(video_folder: str, cam: int) -> str:
    """Return a random mp4 path for the given cam index."""
    folder = Path(video_folder)
    candidates = sorted(folder.glob(f"cam-{cam}_*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No cam-{cam}_*.mp4 files found in {video_folder}")
    return str(random.choice(candidates))


def _extract_first_frame(video_path: str) -> np.ndarray:
    """Extract the first frame of a video using ffmpeg, return as BGR numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting frame from {video_path}:\n{result.stderr[-1000:]}")

    frame = cv2.imread(tmp_path)
    os.unlink(tmp_path)
    if frame is None:
        raise RuntimeError(f"Could not read extracted frame from {tmp_path}")
    return frame


# ---------------------------------------------------------------------------
# Interactive calibration for one camera
# ---------------------------------------------------------------------------

def _calibrate_one_cam(
    frame: np.ndarray,
    cam: int,
    mic_ids: list[int],
    window_name: str,
) -> dict[int, tuple[int, int]]:
    """
    Interactive click-based calibration for one camera frame.
    Returns dict mic_id -> (x, y).

    Controls:
      Left click  - register point
      Right click - undo last point
      ESC         - cancel (raises RuntimeError)
      ENTER/SPACE - confirm (only after all mics clicked)
    """
    if not _has_gui_display():
        # Interactive calibration requires a graphical session.
        raise RuntimeError(
            "No GUI display detected. Run from a graphical session or enable X11 forwarding."
        )

    base = frame.copy()
    mic_points: dict[int, tuple[int, int]] = {}
    click_order: list[int] = []   # ordered list of mic_ids registered so far
    current_step = 0              # index into mic_ids

    def draw() -> np.ndarray:
        # Redraw the current frame with all confirmed points and the next prompt.
        img = base.copy()
        for mid, (x, y) in mic_points.items():
            cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(img, f"M{mid}", (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if current_step < len(mic_ids):
            prompt = f"Cam-{cam} | Click on mic {mic_ids[current_step]}  " \
                     f"({current_step + 1}/{len(mic_ids)})  " \
                     f"| Right-click to undo | ESC to cancel"
        else:
            prompt = f"Cam-{cam} | All {len(mic_ids)} mics registered. " \
                     f"Press ENTER/SPACE to confirm or right-click to undo."

        cv2.rectangle(img, (8, 8), (img.shape[1] - 8, 54), (0, 0, 0), -1)
        cv2.putText(img, prompt, (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def on_mouse(event, x, y, _flags, _param):
        nonlocal current_step
        if event == cv2.EVENT_LBUTTONDOWN and current_step < len(mic_ids):
            # Left click records the next microphone in the configured order.
            mid = mic_ids[current_step]
            mic_points[mid] = (x, y)
            click_order.append(mid)
            current_step += 1

        elif event == cv2.EVENT_RBUTTONDOWN and click_order:
            # Right click removes the most recent point so the operator can correct mistakes.
            last = click_order.pop()
            mic_points.pop(last, None)
            current_step = max(0, current_step - 1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, draw())
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError(f"Calibration for cam-{cam} canceled by user.")

        if current_step >= len(mic_ids) and key in (13, 10, 32):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            raise RuntimeError(f"Calibration window for cam-{cam} closed before completion.")

    cv2.destroyWindow(window_name)
    return mic_points


# ---------------------------------------------------------------------------
# Main calibrator class
# ---------------------------------------------------------------------------

class MicrophoneSurfaceCalibrator:
    """
    Two-camera click-based calibration.

    cam-1 -> mic IDs  1..num_mics_per_cam       (front)
    cam-0 -> mic IDs  num_mics_per_cam+1..total (back)
    """

    def __init__(self, calibration_file: Optional[str] = None):
        self.calibration_file = calibration_file
        self.calibration_data: Optional[CalibrationData] = None

        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _to_payload(self) -> dict:
        cd = self.calibration_data
        if cd is None:
            raise ValueError("No calibration data available.")

        def cam_payload(cc: CamCalibration) -> dict:
            return {
                "cam": cc.cam,
                "mic_ids": cc.mic_ids,
                "microphone_points": [
                    {"id": mid, "x": int(pt[0]), "y": int(pt[1])}
                    for mid, pt in sorted(cc.microphone_points.items())
                ],
            }

        return {
            "num_mics": cd.num_mics,
            "cam0": cam_payload(cd.cam0),
            "cam1": cam_payload(cd.cam1),
        }

    def save_calibration(self, output_path: Optional[str] = None) -> None:
        target = output_path or self.calibration_file
        if target is None:
            raise ValueError("No output path provided.")
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._to_payload(), f, indent=2)
        print(f"Calibration saved -> {path}")

    def load_calibration(self, calibration_path: Optional[str] = None) -> CalibrationData:
        path = calibration_path or self.calibration_file
        if path is None:
            raise ValueError("No calibration file path provided.")

        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        def parse_cam(d: dict) -> CamCalibration:
            pts = {int(item["id"]): (int(item["x"]), int(item["y"]))
                   for item in d["microphone_points"]}
            return CamCalibration(
                cam=int(d["cam"]),
                mic_ids=[int(i) for i in d["mic_ids"]],
                microphone_points=pts,
            )

        self.calibration_data = CalibrationData(
            num_mics=int(payload["num_mics"]),
            cam0=parse_cam(payload["cam0"]),
            cam1=parse_cam(payload["cam1"]),
        )
        return self.calibration_data

    # ------------------------------------------------------------------
    # Calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        video_folder: str,
        num_mics_per_cam: int = 16,
        output_path: Optional[str] = None,
    ) -> CalibrationData:
        """
        Auto-pick a random video for each cam, extract first frame,
        run interactive click calibration for each camera in sequence.

        cam-1 first (front, mic IDs 1..num_mics_per_cam)
        cam-0 second (back,  mic IDs num_mics_per_cam+1..2*num_mics_per_cam)
        """
        total_mics = num_mics_per_cam * 2
        cam1_mic_ids = list(range(1, num_mics_per_cam + 1))
        cam0_mic_ids = list(range(num_mics_per_cam + 1, total_mics + 1))

        # Calibrate the front camera first so the operator can follow a consistent order.
        vid1 = _pick_random_video(video_folder, cam=1)
        print(f"Cam-1 calibration frame from: {Path(vid1).name}")
        frame1 = _extract_first_frame(vid1)
        pts1 = _calibrate_one_cam(frame1, cam=1, mic_ids=cam1_mic_ids,
                                   window_name="Calibration - Cam-1 (front, mics 1..16)")

        # Then calibrate the back camera with the remaining microphone IDs.
        vid0 = _pick_random_video(video_folder, cam=0)
        print(f"Cam-0 calibration frame from: {Path(vid0).name}")
        frame0 = _extract_first_frame(vid0)
        pts0 = _calibrate_one_cam(frame0, cam=0, mic_ids=cam0_mic_ids,
                                   window_name="Calibration - Cam-0 (back, mics 17..32)")

        self.calibration_data = CalibrationData(
            num_mics=total_mics,
            cam0=CamCalibration(cam=0, mic_ids=cam0_mic_ids, microphone_points=pts0),
            cam1=CamCalibration(cam=1, mic_ids=cam1_mic_ids, microphone_points=pts1),
        )

        target = output_path or self.calibration_file
        if target:
            self.save_calibration(target)

        return self.calibration_data

    # ------------------------------------------------------------------
    # Query helpers (used by sync pipeline)
    # ------------------------------------------------------------------

    def get_pixel_coords(self, mic_id: int) -> tuple[int, int]:
        """Return (x, y) pixel coords for mic_id on its camera frame."""
        if self.calibration_data is None:
            raise ValueError("Calibration not loaded.")
        for cc in (self.calibration_data.cam0, self.calibration_data.cam1):
            if mic_id in cc.microphone_points:
                return cc.microphone_points[mic_id]
        raise KeyError(f"Mic ID {mic_id} not found in calibration data.")

    def get_cam_for_mic(self, mic_id: int) -> int:
        """Return which camera (0 or 1) covers mic_id."""
        if self.calibration_data is None:
            raise ValueError("Calibration not loaded.")
        if mic_id in self.calibration_data.cam0.microphone_points:
            return 0
        if mic_id in self.calibration_data.cam1.microphone_points:
            return 1
        raise KeyError(f"Mic ID {mic_id} not found in calibration data.")

    # ------------------------------------------------------------------
    # Highlight (debug / visualization)
    # ------------------------------------------------------------------

    def highlight_microphone(
        self,
        mic_id: int,
        image: Union[str, np.ndarray],
        box_size_px: int = 120,
        thickness: int = 2,
        show: bool = True,
        save: bool = False,
        output_path: Optional[str] = None,
        window_name: str = "Microphone Highlight",
    ) -> np.ndarray:
        """Draw a rectangle centered on mic_id and optionally show/save the result."""
        if self.calibration_data is None:
            raise ValueError("Calibration not loaded.")

        cx, cy = self.get_pixel_coords(mic_id)

        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                raise FileNotFoundError(f"Image not found: {image}")
        else:
            frame = image.copy()

        half = max(1, box_size_px // 2)
        h, w = frame.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w - 1, cx + half)
        y2 = min(h - 1, cy + half)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        cv2.putText(frame, f"Mic {mic_id}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if save:
            # Saving is useful for debugging the projected microphone region offline.
            out = output_path or f"highlight_mic_{mic_id}.png"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out, frame)

        if show:
            if not _has_gui_display():
                raise RuntimeError("No GUI display. Use show=False, save=True.")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, frame)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

        return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Microphone surface calibrator for two-camera setup."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # calibrate
    cal = sub.add_parser("calibrate", help="Run interactive calibration.")
    cal.add_argument("--video-folder", required=True, metavar="PATH",
                     help="Folder containing cam-0_*.mp4 and cam-1_*.mp4 files.")
    cal.add_argument("--output", default="calibration.json", metavar="PATH",
                     help="Output JSON path.")
    cal.add_argument("--num-mics-per-cam", type=int, default=16,
                     help="Number of microphones per camera face.")

    # highlight
    hl = sub.add_parser("highlight", help="Highlight a mic on a frame (debug).")
    hl.add_argument("--calibration", required=True, metavar="PATH")
    hl.add_argument("--mic-id", required=True, type=int)
    hl.add_argument("--video-folder", required=True, metavar="PATH",
                    help="Used to auto-pick a frame for the correct camera.")
    hl.add_argument("--box-size", type=int, default=120)
    hl.add_argument("--save", action="store_true")
    hl.add_argument("--output", default=None, metavar="PATH")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "calibrate":
        cal = MicrophoneSurfaceCalibrator()
        cal.calibrate(
            video_folder=args.video_folder,
            num_mics_per_cam=args.num_mics_per_cam,
            output_path=args.output,
        )

    elif args.command == "highlight":
        cal = MicrophoneSurfaceCalibrator(args.calibration)
        cam = cal.get_cam_for_mic(args.mic_id)
        vid = _pick_random_video(args.video_folder, cam)
        frame = _extract_first_frame(vid)
        cal.highlight_microphone(
            mic_id=args.mic_id,
            image=frame,
            box_size_px=args.box_size,
            show=True,
            save=args.save,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()