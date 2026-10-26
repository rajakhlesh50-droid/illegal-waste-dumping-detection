from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrackingConfig:
    tracker_yaml: str = "bytetrack.yaml"
    person_confidence: float = 0.45
    image_size: int = 512


class PersonTracker:
    """Dedicated ByteTrack/pose wrapper.

    The pose model is expected to be an Ultralytics YOLO pose model.
    """

    def __init__(self, pose_model, config: Optional[TrackingConfig] = None):
        self.pose_model = pose_model
        self.cfg = config or TrackingConfig()

    def update(self, frame) -> List[Dict[str, Any]]:
        persons: List[Dict[str, Any]] = []
        if self.pose_model is None:
            return persons

        try:
            results = self.pose_model.track(
                frame,
                persist=True,
                tracker=self.cfg.tracker_yaml,
                classes=[0],
                conf=self.cfg.person_confidence,
                imgsz=self.cfg.image_size,
                verbose=False,
            )
        except Exception as exc:
            print("Tracking error:", exc)
            return persons

        if not results or results[0].boxes is None:
            return persons

        result = results[0]
        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            confidence = float(boxes.conf[i].cpu().item())
            track_id = None
            if boxes.id is not None:
                try:
                    track_id = int(boxes.id[i].cpu().item())
                except Exception:
                    track_id = None

            wrists = []
            if result.keypoints is not None:
                try:
                    points = result.keypoints.xy[i].cpu().numpy()
                    for index in (9, 10):
                        if len(points) > index:
                            x, y = points[index]
                            if x > 0 and y > 0:
                                wrists.append((int(x), int(y)))
                except Exception:
                    wrists = []

            persons.append({
                "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "confidence": confidence,
                "track_id": track_id,
                "wrists": wrists,
            })

        return persons
