"""
screen_capture.py — Async Screen & Webcam Capture for Aura-NPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides asynchronous screen region capture and webcam frame acquisition.
Uses asyncio executor to prevent UI thread blocking during capture.

Camera feeds run in a dedicated background thread at 30fps.
Frame queue limits memory usage to the most recent N frames.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("aura.screen_capture")

# Optional imports (graceful degradation)
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logger.warning("mss not installed. Install: pip install mss")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. Install: pip install opencv-python")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ScreenCapture:
    """
    Async-capable screen region and webcam frame capture.

    Usage:
        capture = ScreenCapture()
        image = await capture.capture_screen_async()   # PIL.Image
        frame = await capture.get_webcam_frame_async() # numpy array
    """

    def __init__(
        self,
        monitor_index: int = 1,
        capture_region: Optional[dict] = None,
        webcam_index: int = 0,
        frame_queue_size: int = 5,
    ):
        """
        Args:
            monitor_index:   Monitor to capture (1 = primary)
            capture_region:  Dict with top/left/width/height or None for full screen
            webcam_index:    OpenCV camera device index
            frame_queue_size: Max frames to buffer in the webcam queue
        """
        self.monitor_index = monitor_index
        self.capture_region = capture_region
        self.webcam_index = webcam_index
        self._frame_queue: deque = deque(maxlen=frame_queue_size)
        self._webcam: Optional[Any] = None
        self._webcam_running: bool = False
        self._capture_count: int = 0

    # ── Screen Capture ────────────────────────────────────────────────────────

    async def capture_screen_async(self) -> Optional[Any]:
        """
        Asynchronous full-screen capture.
        Runs mss capture in thread pool to avoid blocking the NiceGUI event loop.

        Returns: PIL.Image or numpy array, or None on failure.
        """
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, self._capture_screen_sync)
        self._capture_count += 1
        return image

    def _capture_screen_sync(self) -> Optional[Any]:
        """Synchronous screen capture using mss."""
        if not MSS_AVAILABLE:
            return self._generate_dummy_image()

        try:
            with mss.mss() as sct:
                if self.capture_region:
                    monitor = self.capture_region
                else:
                    monitor = sct.monitors[self.monitor_index]

                screenshot = sct.grab(monitor)
                img_array = np.array(screenshot)

                # mss returns BGRA — convert to RGB
                img_rgb = img_array[:, :, [2, 1, 0]]  # BGR → RGB (drop alpha)

                if PIL_AVAILABLE:
                    return PILImage.fromarray(img_rgb)
                return img_rgb

        except Exception as exc:
            logger.error("Screen capture failed: %s", exc)
            return self._generate_dummy_image()

    async def capture_region_async(
        self,
        top: int,
        left: int,
        width: int,
        height: int,
    ) -> Optional[Any]:
        """Capture a specific screen region (e.g., a textbook diagram)."""
        region = {"top": top, "left": left, "width": width, "height": height}
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._capture_region_sync(region)
        )

    def _capture_region_sync(self, region: dict) -> Optional[Any]:
        """Synchronous region capture."""
        if not MSS_AVAILABLE:
            return self._generate_dummy_image(region.get("width", 400), region.get("height", 300))

        try:
            with mss.mss() as sct:
                screenshot = sct.grab(region)
                img_array = np.array(screenshot)
                img_rgb = img_array[:, :, [2, 1, 0]]
                if PIL_AVAILABLE:
                    return PILImage.fromarray(img_rgb)
                return img_rgb
        except Exception as exc:
            logger.error("Region capture failed: %s", exc)
            return None

    # ── Webcam Feed ───────────────────────────────────────────────────────────

    async def get_webcam_frame_async(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame from the webcam feed.
        Returns None if webcam is not running or no frame available.
        """
        if not CV2_AVAILABLE:
            return self._generate_dummy_image()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._grab_webcam_frame)

    def _grab_webcam_frame(self) -> Optional[np.ndarray]:
        """Grab a single frame from the webcam."""
        if self._frame_queue:
            return self._frame_queue[-1]

        # Try to open webcam on demand
        try:
            cap = cv2.VideoCapture(self.webcam_index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.debug("Webcam frame grab failed: %s", exc)
        return None

    def start_webcam_stream(self) -> None:
        """Start continuous webcam capture in a background thread."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available — webcam stream disabled.")
            return

        import threading
        self._webcam_running = True
        thread = threading.Thread(target=self._webcam_loop, daemon=True)
        thread.start()
        logger.info("Webcam stream started on device %d", self.webcam_index)

    def stop_webcam_stream(self) -> None:
        """Signal the webcam thread to stop."""
        self._webcam_running = False

    def _webcam_loop(self) -> None:
        """Background thread: continuously capture and enqueue frames."""
        try:
            cap = cv2.VideoCapture(self.webcam_index)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self._webcam_running:
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._frame_queue.append(rgb_frame)
                else:
                    time.sleep(0.033)

            cap.release()
        except Exception as exc:
            logger.error("Webcam loop error: %s", exc)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _generate_dummy_image(
        self,
        width: int = 896,
        height: int = 896,
    ) -> Any:
        """
        Generate a synthetic test image when hardware capture is unavailable.
        Used for demo/testing without a real screen or camera.
        """
        # Create a cyberpunk-ish gradient test image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Add AMD color gradient
        for y in range(height):
            r = int(255 * y / height)
            img[y, :, 0] = r          # Red channel gradient
            img[y, :, 1] = 26         # Dark green (#1A)
            img[y, :, 2] = 0

        # Add a simulated diagram label
        if CV2_AVAILABLE:
            cv2.putText(
                img,
                "AURA-NPU Demo Capture",
                (50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 102, 0),
                2,
            )

        if PIL_AVAILABLE:
            return PILImage.fromarray(img)
        return img

    def save_frame(self, image: Any, output_dir: str = ".") -> Optional[Path]:
        """Save a captured frame to disk (for debugging)."""
        try:
            ts = int(time.time() * 1000)
            out_path = Path(output_dir) / f"aura_capture_{ts}.png"

            if PIL_AVAILABLE and hasattr(image, "save"):
                image.save(out_path, format="PNG")
            elif isinstance(image, np.ndarray):
                if CV2_AVAILABLE:
                    cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            return out_path
        except Exception as exc:
            logger.error("Frame save failed: %s", exc)
            return None

    @property
    def capture_count(self) -> int:
        """Total number of successful captures in this session."""
        return self._capture_count
