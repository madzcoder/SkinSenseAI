#!/usr/bin/env python3
"""
SkinSense AI — Raspberry Pi 4 Backend
Hardware: Nextion NX4827T043 (UART) + Camera Module 3 NoIR + Pi 4 8GB
Protocol: Nextion Serial @ 9600 baud via /dev/ttyS0
"""

import serial
import time
import threading
import struct
import numpy as np
import cv2
import logging
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2, Preview

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("SkinSense")

# ─── Config ───────────────────────────────────────────────────────────────────
UART_PORT    = "/dev/ttyS0"
UART_BAUD    = 9600
CAPTURE_DIR  = Path("/home/pi/skinsense/captures")
NEXTION_TERM = bytes([0xFF, 0xFF, 0xFF])   # Command terminator

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# NEXTION DISPLAY DRIVER
# ════════════════════════════════════════════════════════════════════════════════
class NextionDisplay:
    """
    Handles all serial communication with the Nextion NX4827T043.
    Pages and component IDs must match your .HMI design.

    Page layout (design these in Nextion Editor):
      0 = welcome      → b0 (Start button)
      1 = prep         → b0 (Ready), b1 (Back)
      2 = position     → b0 (Open Camera), b1 (Back), t0 (hint text)
      3 = capture      → b0 (Scan), b1 (Back), x0 (status bar), p0 (preview crop)
      4 = analyzing    → j0 (progress bar), t0 (stage label)
      5 = result       → t0 (risk), t1 (score), b0 (New Scan), b1 (Save)
    """

    def __init__(self, port=UART_PORT, baud=UART_BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        log.info(f"Nextion connected on {port} @ {baud}")
        time.sleep(0.2)

    def send(self, cmd: str):
        """Send a raw Nextion command."""
        data = cmd.encode("ascii") + NEXTION_TERM
        self.ser.write(data)
        time.sleep(0.02)

    def goto_page(self, page: int):
        self.send(f"page {page}")
        log.info(f"→ Page {page}")

    def set_text(self, component: str, value: str):
        self.send(f'{component}.txt="{value}"')

    def set_val(self, component: str, value: int):
        self.send(f"{component}.val={value}")

    def set_color(self, component: str, color: int):
        """Color as 16-bit 565 integer. e.g. teal=0x07E0, red=0xF800"""
        self.send(f"{component}.bco={color}")

    def set_progress(self, component: str, value: int):
        """Update a progress bar (0-100)."""
        self.set_val(component, max(0, min(100, value)))

    def show_component(self, component: str, visible: bool):
        self.send(f"vis {component},{1 if visible else 0}")

    def read_event(self) -> dict | None:
        """
        Non-blocking read of touch events from Nextion.
        Returns: {'page': int, 'component': int, 'event': 'press'|'release'} or None
        """
        if self.ser.in_waiting >= 7:
            raw = self.ser.read(7)
            if raw[0] == 0x65:
                return {
                    "page": raw[1],
                    "component": raw[2],
                    "event": "press" if raw[3] == 0x01 else "release"
                }
        return None

    def refresh(self):
        self.send("ref 0")   # Refresh all

    def close(self):
        self.ser.close()


# ════════════════════════════════════════════════════════════════════════════════
# CAMERA CONTROLLER
# ════════════════════════════════════════════════════════════════════════════════
class SkinCamera:
    """
    Manages the Camera Module 3 NoIR via picamera2.
    Captures full-res stills and streams low-res frames for alignment detection.
    """

    STREAM_RES  = (640, 480)    # For live alignment check
    CAPTURE_RES = (4608, 2592)  # Camera 3 native res

    def __init__(self):
        self.cam = Picamera2()

        # Preview config (low-res, fast) ─ used during positioning
        self.preview_cfg = self.cam.create_preview_configuration(
            main={"size": self.STREAM_RES, "format": "RGB888"},
            controls={"FrameRate": 15, "AwbEnable": True, "AeEnable": True}
        )

        # Still config (high-res) ─ used for AI capture
        self.still_cfg = self.cam.create_still_configuration(
            main={"size": self.CAPTURE_RES, "format": "RGB888"},
            controls={"AwbEnable": False, "AeEnable": False}   # Lock exposure for capture
        )

        self.cam.configure(self.preview_cfg)
        self.cam.start()
        log.info("Camera Module 3 NoIR started")
        time.sleep(1)  # Warm-up

    def grab_frame(self) -> np.ndarray:
        """Return a BGR frame for analysis."""
        frame = self.cam.capture_array("main")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def capture_still(self) -> Path:
        """
        Switch to high-res still, capture, switch back to preview.
        Returns path to saved JPEG.
        """
        log.info("Capturing high-res still...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CAPTURE_DIR / f"lesion_{ts}.jpg"

        self.cam.switch_mode_and_capture_file(self.still_cfg, str(path), signal_function=None)
        log.info(f"Saved: {path}")
        return path

    def close(self):
        self.cam.stop()
        self.cam.close()


# ════════════════════════════════════════════════════════════════════════════════
# ALIGNMENT DETECTOR
# ════════════════════════════════════════════════════════════════════════════════
class AlignmentDetector:
    """
    Analyses live frames to guide the patient.
    Uses classical CV (no heavy ML needed for this step).
    """

    # Target zone: centre circle of the 640x480 preview
    TARGET_CX, TARGET_CY = 320, 240
    TARGET_R = 80   # pixels

    @staticmethod
    def check_frame(frame: np.ndarray) -> dict:
        """
        Returns alignment status and feedback message.
        status: 'good' | 'no_lesion' | 'off_center' | 'too_dark' | 'too_close'
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        roi = frame[cy-100:cy+100, cx-100:cx+100]

        # ── Brightness check ─────────────────────────────────────────────────
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        if mean_brightness < 40:
            return {"status": "too_dark", "msg": "Area too dark — add more light"}

        # ── Detect a skin-like region (simple HSV threshold) ──────────────────
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 20, 60), (25, 180, 255))
        skin_ratio = skin_mask.sum() / (skin_mask.size * 255)

        if skin_ratio < 0.15:
            return {"status": "no_lesion", "msg": "Place skin lesion in the circle"}

        # ── Edge density to detect if too close (blurry) ─────────────────────
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.size * 255)

        if edge_density > 0.35:
            return {"status": "too_close", "msg": "Move slightly further away"}

        # ── Check if a dark region (lesion candidate) is centred ─────────────
        dark_mask = cv2.inRange(gray, 0, 80)
        moments = cv2.moments(dark_mask)
        if moments["m00"] > 500:
            lx = int(moments["m10"] / moments["m00"])
            ly = int(moments["m01"] / moments["m00"])
            offset = ((lx - 100)**2 + (ly - 100)**2) ** 0.5
            if offset > 40:
                return {"status": "off_center", "msg": f"Move {'left' if lx>100 else 'right'} — centre lesion"}

        return {"status": "good", "msg": "✓ Perfect — hold still!"}

    @staticmethod
    def compute_thumbnail_for_nextion(frame: np.ndarray, size=(160, 120)) -> bytes:
        """Resize and encode frame as JPEG for Nextion picture display."""
        thumb = cv2.resize(frame, size)
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()


# ════════════════════════════════════════════════════════════════════════════════
# DUMMY AI ANALYSER (replace with your real model)
# ════════════════════════════════════════════════════════════════════════════════
class SkinAI:
    """
    Stub for ABCD dermoscopy analysis.
    Plug in TFLite / ONNX / cloud API here.
    """

    def analyse(self, image_path: Path, progress_cb=None) -> dict:
        stages = [
            "Preprocessing image",
            "Detecting boundaries",
            "Analysing pigmentation",
            "Running AI model",
            "Generating report",
        ]
        for i, stage in enumerate(stages):
            log.info(f"AI: {stage}...")
            if progress_cb:
                progress_cb(int((i / len(stages)) * 100), stage)
            time.sleep(0.8)   # Replace with real inference

        if progress_cb:
            progress_cb(100, "Done")

        # ── Return dummy result ───────────────────────────────────────────────
        return {
            "risk":      "Low Risk",
            "score":     92,
            "color":     0x07E0,   # green in 565
            "asymmetry": "Low",
            "border":    "Regular",
            "color_var": "Uniform",
            "diameter":  "4.2 mm",
            "note":      "No urgent concern. Monitor monthly.",
        }


# ════════════════════════════════════════════════════════════════════════════════
# MAIN STATE MACHINE
# ════════════════════════════════════════════════════════════════════════════════
class SkinSenseApp:
    """
    Coordinates Nextion UI ↔ Camera ↔ AI.

    Touch event routing (page / component):
      Page 0 / B0  → PrepScreen
      Page 1 / B0  → PositionScreen    | B1 → WelcomeScreen
      Page 2 / B0  → CaptureScreen     | B1 → PrepScreen
      Page 3 / B0  → trigger scan      | B1 → PositionScreen
      Page 5 / B0  → WelcomeScreen (new scan)
      Page 5 / B1  → save report
    """

    def __init__(self):
        self.display = NextionDisplay()
        self.camera  = SkinCamera()
        self.ai      = SkinAI()
        self.detector = AlignmentDetector()
        self.running = True

        self._alignment_status = "idle"
        self._capture_thread   = None
        self._align_thread     = None

    # ── Navigation helpers ────────────────────────────────────────────────────
    def show_welcome(self):
        self.display.goto_page(0)

    def show_prep(self):
        self.display.goto_page(1)

    def show_position(self):
        self.display.goto_page(2)
        self.display.set_text("t0", "Align the lesion inside the circle")

    def show_capture(self):
        self.display.goto_page(3)
        self._start_alignment_loop()

    def show_analyzing(self):
        self._stop_alignment_loop()
        self.display.goto_page(4)
        self.display.set_progress("j0", 0)
        self.display.set_text("t0", "Starting analysis...")

    def show_result(self, result: dict):
        self.display.goto_page(5)
        self.display.set_text("t0", result["risk"])
        self.display.set_text("t1", f"{result['score']}/100")
        self.display.set_text("t2", result["note"])
        self.display.set_color("t0", result["color"])

    # ── Alignment feedback loop ───────────────────────────────────────────────
    def _start_alignment_loop(self):
        self._alignment_running = True
        self._align_thread = threading.Thread(target=self._alignment_loop, daemon=True)
        self._align_thread.start()

    def _stop_alignment_loop(self):
        self._alignment_running = False

    def _alignment_loop(self):
        STATUS_COLORS = {
            "good":      0x07E0,   # green
            "no_lesion": 0xFFE0,   # yellow
            "off_center":0xFD20,   # orange
            "too_dark":  0x8410,   # grey
            "too_close": 0xF800,   # red
        }
        last_status = None
        while self._alignment_running:
            try:
                frame = self.camera.grab_frame()
                result = self.detector.check_frame(frame)
                status = result["status"]
                msg    = result["msg"]

                if status != last_status:
                    color = STATUS_COLORS.get(status, 0xFFFF)
                    self.display.set_text("x0", msg)
                    self.display.set_color("x0", color)
                    self._alignment_status = status
                    last_status = status

            except Exception as e:
                log.warning(f"Alignment loop error: {e}")
            time.sleep(0.15)

    # ── Scan trigger ─────────────────────────────────────────────────────────
    def trigger_scan(self):
        if self._alignment_status != "good":
            self.display.set_text("x0", "⚠ Adjust position first")
            return

        self._capture_thread = threading.Thread(target=self._scan_pipeline, daemon=True)
        self._capture_thread.start()

    def _scan_pipeline(self):
        # 3-second countdown on Nextion
        for i in range(3, 0, -1):
            self.display.set_text("x0", f"Capturing in {i}...")
            time.sleep(1)

        self.display.set_text("x0", "📸 Capturing!")
        image_path = self.camera.capture_still()

        # Switch to analysing screen
        self.show_analyzing()

        def on_progress(pct, stage):
            self.display.set_progress("j0", pct)
            self.display.set_text("t0", stage)

        result = self.ai.analyse(image_path, progress_cb=on_progress)
        self.show_result(result)

    # ── Touch event dispatcher ────────────────────────────────────────────────
    def handle_touch(self, event: dict):
        page = event["page"]
        comp = event["component"]
        evt  = event["event"]

        if evt != "press":
            return

        log.info(f"Touch: page={page} component={comp}")

        routes = {
            (0, 0): self.show_prep,
            (1, 0): self.show_position,
            (1, 1): self.show_welcome,
            (2, 0): self.show_capture,
            (2, 1): self.show_prep,
            (3, 0): self.trigger_scan,
            (3, 1): lambda: (self._stop_alignment_loop(), self.show_position()),
            (5, 0): self.show_welcome,
            (5, 1): self._save_report,
        }

        action = routes.get((page, comp))
        if action:
            action()
        else:
            log.warning(f"Unhandled touch: page={page} comp={comp}")

    def _save_report(self):
        log.info("Saving report to USB/SD...")
        # TODO: implement PDF export or send to server
        self.display.set_text("t2", "Report saved ✓")

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        log.info("SkinSense AI starting...")
        self.show_welcome()

        try:
            while self.running:
                event = self.display.read_event()
                if event:
                    self.handle_touch(event)
                time.sleep(0.02)
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self._stop_alignment_loop()
            self.camera.close()
            self.display.close()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SkinSenseApp()
    app.run()
