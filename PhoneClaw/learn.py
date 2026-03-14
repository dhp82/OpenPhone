"""Learning mode for PhoneClaw — record human demonstrations and extract lessons.

How it works
------------
1. DemoRecorder polls the device screen at ~8 fps using WDA.
2. Each frame pair is compared.  When a significant change is detected:
   a. The diff image is analysed with OpenCV HoughCircles to locate the iOS
      "Show Touches" indicator (a semi-transparent circle that appears at the
      tap point when Settings → Developer → Show Touches is enabled).
   b. If no circle is found the centroid of the largest changed region is used
      as a fallback estimate.
3. Each changed frame plus its estimated tap coordinate is saved to disk.
4. After recording ends the VLM is called for each frame to extract reusable
   navigation lessons which are stored in the ExperienceLog.

Prerequisites for best results
-------------------------------
Enable "Show Touches" on the iOS device before starting the demo:

  iOS 16+:  Settings → Privacy & Security → Developer Mode → Show Touches
  Older:    Settings → Accessibility → Touch → Show Touches

With "Show Touches" active, every tap leaves a white-circle overlay visible
in WDA screenshots.  HoughCircles detects the circle and returns the exact
tap centre.  Without it the module still works but falls back to the centre
of the changed screen region (less precise for large animations).
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from PhoneClaw.screenshot import get_screenshot

DEFAULT_DEMO_BASE = Path(__file__).parent / "data" / "demos"

# ---------------------------------------------------------------------------
# HoughCircles parameters — tuned for typical iOS Retina screenshots.
# The touch indicator is a semi-transparent circle roughly 25–40 logical pts
# in diameter; at 3× scale that is ~75–120 px, but WDA returns screenshots
# at logical resolution (typically 390–430 pt wide), so effective radii are
# 12–50 px.
# ---------------------------------------------------------------------------
_HOUGH_DP = 1.5         # inverse ratio of accumulator resolution
_HOUGH_MIN_DIST = 40    # minimum distance between detected circle centres
_HOUGH_PARAM1 = 60      # Canny edge upper threshold applied to diff image
_HOUGH_PARAM2 = 16      # accumulator threshold — lower = more permissive
_HOUGH_MIN_R = 12       # minimum radius in px
_HOUGH_MAX_R = 55       # maximum radius in px

# Pixel intensity threshold for the abs-diff map (0-255).
_DIFF_PIXEL_THRESHOLD = 20

# Default minimum fraction of pixels that must change to count as an event.
_CHANGE_THRESHOLD_DEFAULT = 0.003   # 0.3 %


# ---------------------------------------------------------------------------
# Frame data
# ---------------------------------------------------------------------------

@dataclass
class DemoFrame:
    """One recorded screen-change event."""
    idx: int
    timestamp: float
    screenshot_b64: str     # annotated screenshot (tap marked with red circle)
    width: int
    height: int
    tap_x_rel: Optional[float]   # normalised [0, 1], None if not detected
    tap_y_rel: Optional[float]
    detection_method: str        # "hough_circles" | "diff_centroid" | "none"
    change_pct: float            # fraction of pixels that changed [0, 1]


# ---------------------------------------------------------------------------
# Touch indicator detection
# ---------------------------------------------------------------------------

def _b64_to_gray(b64: str) -> np.ndarray:
    """Decode a base-64 PNG and return a grayscale numpy array."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode screenshot image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _compute_diff(
    gray_before: np.ndarray, gray_after: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return (abs_diff_image, fraction_of_changed_pixels)."""
    diff = cv2.absdiff(gray_before, gray_after)
    _, binary = cv2.threshold(diff, _DIFF_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
    change_pct = float(np.count_nonzero(binary)) / binary.size
    return diff, change_pct


def _detect_tap(
    diff: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[Optional[float], Optional[float], str]:
    """Estimate the tap position from an abs-diff frame.

    Method 1 (preferred): HoughCircles on a Gaussian-blurred diff image.
        The "Show Touches" overlay manifests as a roughly circular bright
        region in the diff map.  When detected we return the circle centre.

    Method 2 (fallback): Centroid of the largest connected changed component.
        Works even without Show Touches, but is less precise when a large
        area of the UI changes (e.g. a page transition animation).

    Returns:
        (rel_x, rel_y, method_name) — coordinates in [0, 1] relative to the
        image dimensions, or (None, None, "none") if localisation failed.
    """
    # ── Method 1: HoughCircles on diff ────────────────────────────────────
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        diff_blur,
        cv2.HOUGH_GRADIENT,
        dp=_HOUGH_DP,
        minDist=_HOUGH_MIN_DIST,
        param1=_HOUGH_PARAM1,
        param2=_HOUGH_PARAM2,
        minRadius=_HOUGH_MIN_R,
        maxRadius=_HOUGH_MAX_R,
    )

    if circles is not None:
        # Among all candidate circles, pick the one with the largest radius
        # (most likely to be the finger-down indicator rather than UI noise).
        best = max(circles[0], key=lambda c: c[2])
        return float(best[0]) / img_w, float(best[1]) / img_h, "hough_circles"

    # ── Method 2: Centroid of largest changed region ───────────────────────
    _, binary = cv2.threshold(diff, _DIFF_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return None, None, "none"

    # Label 0 is background; find the largest foreground component.
    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[best_label]
    return float(cx) / img_w, float(cy) / img_h, "diff_centroid"


# ---------------------------------------------------------------------------
# DemoRecorder
# ---------------------------------------------------------------------------

class DemoRecorder:
    """Records a human demonstration on the device and extracts lessons.

    Typical usage::

        recorder = DemoRecorder(
            wda_url="http://localhost:8100",
            session_id="abc123",
            app_name="美团",
            task_description="查看历史订单",
            experience=exp_log,
        )
        recorder.start()
        input("\\nPerform the demo on the device, then press Enter to stop...\\n")
        recorder.stop()
        lessons = recorder.analyze_and_learn(agent=exec_agent)
        print(recorder.summary())
    """

    def __init__(
        self,
        wda_url: str,
        session_id: str,
        app_name: str,
        task_description: str,
        demo_dir: Optional[Path] = None,
        poll_interval: float = 0.12,               # seconds between polls (~8 fps)
        change_threshold: float = _CHANGE_THRESHOLD_DEFAULT,
        experience=None,                            # Optional[ExperienceLog]
    ):
        self.wda_url = wda_url
        self.session_id = session_id
        self.app_name = app_name
        self.task_description = task_description
        self.poll_interval = poll_interval
        self.change_threshold = change_threshold
        self.experience = experience

        ts = int(time.time())
        safe_app = app_name.replace(" ", "_").replace("/", "_")[:20]
        self.demo_dir: Path = demo_dir or (DEFAULT_DEMO_BASE / f"{safe_app}_{ts}")
        self.demo_dir.mkdir(parents=True, exist_ok=True)

        self.frames: list[DemoFrame] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_idx: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background polling in a daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="demo-recorder", daemon=True
        )
        self._thread.start()
        print(
            f"\n[Learn] Recording started"
            f"\n        App  : {self.app_name}"
            f"\n        Task : {self.task_description}"
            f"\n        Dir  : {self.demo_dir}"
            f"\n"
            f"\n[Learn] TIP — enable 'Show Touches' on the device for precise"
            f"\n        tap detection: Settings → Developer → Show Touches\n"
        )

    def stop(self) -> None:
        """Stop polling and wait for the background thread to exit."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        print(
            f"\n[Learn] Recording stopped — {len(self.frames)} event(s) captured."
        )

    # ------------------------------------------------------------------
    # Background polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Grab screenshots continuously and emit events on changes."""
        while not self._stop_event.is_set():
            try:
                shot = get_screenshot(
                    wda_url=self.wda_url,
                    session_id=self.session_id,
                )
                try:
                    gray = _b64_to_gray(shot.base64_data)
                except Exception:
                    time.sleep(self.poll_interval)
                    continue

                if self._prev_gray is not None:
                    diff, change_pct = _compute_diff(self._prev_gray, gray)
                    if change_pct >= self.change_threshold:
                        tap_x, tap_y, method = _detect_tap(
                            diff, shot.width, shot.height
                        )
                        self._on_event(
                            b64=shot.base64_data,
                            width=shot.width,
                            height=shot.height,
                            tap_x=tap_x,
                            tap_y=tap_y,
                            method=method,
                            change_pct=change_pct,
                        )

                self._prev_gray = gray

            except Exception:
                pass  # transient WDA errors — keep polling

            time.sleep(self.poll_interval)

    def _on_event(
        self,
        b64: str,
        width: int,
        height: int,
        tap_x: Optional[float],
        tap_y: Optional[float],
        method: str,
        change_pct: float,
    ) -> None:
        """Handle a detected screen-change event."""
        self._frame_idx += 1
        idx = self._frame_idx

        # Draw a red circle marker at the detected tap position
        annotated = (
            _annotate_tap(b64, tap_x, tap_y, width, height)
            if tap_x is not None
            else b64
        )

        frame = DemoFrame(
            idx=idx,
            timestamp=time.time(),
            screenshot_b64=annotated,
            width=width,
            height=height,
            tap_x_rel=tap_x,
            tap_y_rel=tap_y,
            detection_method=method,
            change_pct=change_pct,
        )
        self.frames.append(frame)

        # Persist frame to disk immediately
        _save_b64_png(annotated, self.demo_dir / f"frame_{idx:04d}.png")

        pct_str = f"{change_pct * 100:.1f}%"
        coord_str = (
            f"tap≈({tap_x:.3f}, {tap_y:.3f}) [{method}]"
            if tap_x is not None
            else "tap: not detected"
        )
        print(f"[Learn] Frame {idx:3d}  Δ{pct_str:>6}  {coord_str}")

    # ------------------------------------------------------------------
    # VLM analysis
    # ------------------------------------------------------------------

    def analyze_and_learn(self, agent) -> list[str]:
        """Send each recorded frame to the VLM and store the extracted lessons.

        For each frame the VLM receives:
          - The annotated screenshot (tap marker drawn in red)
          - The detected tap coordinate (normalised %)
          - App name, task description, step number, change magnitude

        Returns a flat list of lesson descriptions that were added to the
        ExperienceLog (empty list when no experience object is provided).
        """
        if not self.frames:
            print("[Learn] No frames to analyse.")
            return []

        from PhoneClaw.prompts import (
            DEMO_ANALYSIS_SYSTEM_PROMPT,
            DEMO_ANALYSIS_USER_TEMPLATE,
        )

        all_lessons: list[str] = []
        total = len(self.frames)
        print(f"\n[Learn] Analysing {total} frame(s) with VLM...")

        for frame in self.frames:
            tap_x_pct = (
                f"{frame.tap_x_rel * 100:.1f}" if frame.tap_x_rel is not None else "?"
            )
            tap_y_pct = (
                f"{frame.tap_y_rel * 100:.1f}" if frame.tap_y_rel is not None else "?"
            )

            if frame.detection_method == "hough_circles":
                detection_note = (
                    " (precise — detected via Show Touches indicator)"
                )
            elif frame.detection_method == "diff_centroid":
                detection_note = (
                    " (approximate — estimated from changed region centroid;"
                    " enable Show Touches for better accuracy)"
                )
            else:
                detection_note = (
                    " (unknown — large UI transition, tap position not localised)"
                )

            user_text = DEMO_ANALYSIS_USER_TEMPLATE.format(
                app_name=self.app_name,
                task_description=self.task_description,
                step_num=frame.idx,
                total_steps=total,
                tap_x_pct=tap_x_pct,
                tap_y_pct=tap_y_pct,
                detection_note=detection_note,
                change_pct=round(frame.change_pct * 100, 1),
            )

            messages = [
                {"role": "system", "content": DEMO_ANALYSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/png;base64,{frame.screenshot_b64}"
                                )
                            },
                        },
                    ],
                },
            ]

            try:
                raw = agent.act(messages)
                lessons = _parse_lessons(raw)
            except Exception as exc:
                print(f"[Learn]   Frame {frame.idx}: VLM error — {exc}")
                continue

            added = 0
            for lesson in lessons:
                desc = lesson.get("description", "").strip()
                if not desc:
                    continue
                all_lessons.append(desc)
                if self.experience is not None:
                    stored = self.experience.add_lesson(
                        app=lesson.get("app") or self.app_name,
                        lesson_type=lesson.get("lesson_type", "ui_knowledge"),
                        description=desc,
                        source_task=self.task_description,
                        confidence=lesson.get("confidence", "medium"),
                    )
                    if stored:
                        added += 1

            print(
                f"[Learn]   Frame {frame.idx}: "
                f"{len(lessons)} lesson(s) extracted, {added} new."
            )

        if self.experience is not None:
            self.experience.save()
            # Auto-compact if the demo pushed any app over the threshold
            self.experience.compact_if_needed(agent)

        # Persist a summary JSON alongside the frames
        _save_summary(self.demo_dir / "demo_summary.json", self, all_lessons)

        print(
            f"\n[Learn] Done — {len(all_lessons)} lesson(s) extracted and "
            f"saved to ExperienceLog."
        )
        return all_lessons

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        detected = sum(1 for f in self.frames if f.tap_x_rel is not None)
        method_counts: dict[str, int] = {}
        for f in self.frames:
            method_counts[f.detection_method] = (
                method_counts.get(f.detection_method, 0) + 1
            )
        method_str = "  ".join(
            f"{m}: {n}" for m, n in sorted(method_counts.items())
        )
        return (
            f"\nDemo recording summary:\n"
            f"  App              : {self.app_name}\n"
            f"  Task             : {self.task_description}\n"
            f"  Frames captured  : {len(self.frames)}\n"
            f"  Tap detected     : {detected}/{len(self.frames)}\n"
            f"  Detection methods: {method_str or 'n/a'}\n"
            f"  Output dir       : {self.demo_dir}\n"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _annotate_tap(
    b64: str,
    rel_x: float,
    rel_y: float,
    width: int,
    height: int,
    radius: int = 22,
) -> str:
    """Overlay a red circle on the screenshot at the detected tap location.

    Returns the annotated screenshot as base-64 PNG, or the original on error.
    """
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        px = int(rel_x * width)
        py = int(rel_y * height)
        # Outer ring
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            outline=(255, 50, 50, 230),
            width=4,
        )
        # Centre dot
        dot_r = 6
        draw.ellipse(
            [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
            fill=(255, 50, 50, 200),
        )
        annotated = Image.alpha_composite(img, overlay).convert("RGB")
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return b64


def _save_b64_png(b64: str, path: Path) -> None:
    try:
        path.write_bytes(base64.b64decode(b64))
    except Exception:
        pass


def _parse_lessons(raw: str) -> list[dict]:
    """Extract a JSON array from a raw VLM response string."""
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        return json.loads(raw[start: end + 1])
    except json.JSONDecodeError:
        return []


def _save_summary(path: Path, recorder: DemoRecorder, lessons: list[str]) -> None:
    data = {
        "app": recorder.app_name,
        "task": recorder.task_description,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames_total": len(recorder.frames),
        "lessons_extracted": len(lessons),
        "lessons": lessons,
        "frames": [
            {
                "idx": f.idx,
                "tap_x_rel": f.tap_x_rel,
                "tap_y_rel": f.tap_y_rel,
                "detection_method": f.detection_method,
                "change_pct": round(f.change_pct * 100, 2),
            }
            for f in recorder.frames
        ],
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
