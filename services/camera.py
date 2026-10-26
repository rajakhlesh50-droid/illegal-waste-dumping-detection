from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2


@dataclass
class CameraConfig:
    mode: str = "PHONE"  # PHONE or PC
    phone_url: str = "http://10.124.14.213:8080/video"
    pc_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 15
    buffer_size: int = 1
    read_failure_limit: int = 30
    reconnect_delay: float = 0.5


class CameraManager:
    """Thread-safe camera/phone stream manager with reconnect support."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.cfg = config or CameraConfig()
        self._cap = None
        self._lock = threading.Lock()
        self._failures = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _open_capture(self):
        if self.cfg.mode.upper() == "PHONE":
            if not self.cfg.phone_url.startswith(("http://", "https://")):
                raise ValueError("phone_url must start with http:// or https://")
            print("Opening PHONE camera:", self.cfg.phone_url)
            cap = cv2.VideoCapture(self.cfg.phone_url)
        else:
            print("Opening PC camera:", self.cfg.pc_index)
            cap = cv2.VideoCapture(self.cfg.pc_index, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                print("DirectShow failed; trying default backend...")
                cap = cv2.VideoCapture(self.cfg.pc_index)

        if cap is None or not cap.isOpened():
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            return None

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.buffer_size)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        return cap

    def open(self) -> bool:
        with self._lock:
            if self.is_open:
                return True
            try:
                self._cap = self._open_capture()
            except Exception as exc:
                print("Camera open error:", exc)
                self._cap = None
            self._failures = 0
            if self.is_open:
                print("Camera opened successfully")
                return True
            print("ERROR: Camera could not be opened")
            return False

    def read(self) -> Tuple[bool, Optional[Any]]:
        if not self.is_open and not self.open():
            time.sleep(self.cfg.reconnect_delay)
            return False, None

        success, frame = self._cap.read()
        if success and frame is not None:
            self._failures = 0
            return True, frame

        self._failures += 1
        print(f"Camera read failed ({self._failures}/{self.cfg.read_failure_limit})")
        if self._failures >= self.cfg.read_failure_limit:
            self.reconnect()
        return False, None

    def reconnect(self) -> bool:
        with self._lock:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
            self._failures = 0
        time.sleep(self.cfg.reconnect_delay)
        return self.open()

    def frames(self):
        while True:
            success, frame = self.read()
            if not success:
                continue
            yield frame

    def stop(self) -> None:
        with self._lock:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
            self._failures = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
