from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Optional

import cv2


@dataclass
class EvidenceConfig:
    output_dir: str = "static/results"
    camera_id: str = "CAM-01"
    location: str = "Unknown"
    jpeg_quality: int = 90


class EvidenceManager:
    """Creates verified evidence images and returns a Flask evidence route."""

    def __init__(self, config: Optional[EvidenceConfig] = None):
        self.cfg = config or EvidenceConfig()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.last_evidence_url: Optional[str] = None

    def create(self, frame, persons: Iterable[dict], wastes: Iterable[dict], title: str = "ILLEGAL DUMPING CONFIRMED") -> Optional[str]:
        if frame is None:
            return None

        try:
            evidence = frame.copy()
            height, width = evidence.shape[:2]

            cv2.rectangle(evidence, (0, 0), (width, 110), (0, 0, 200), -1)
            cv2.putText(evidence, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(evidence, f"Camera: {self.cfg.camera_id}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            cv2.putText(evidence, f"Location: {self.cfg.location}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            cv2.putText(evidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (20, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            for person in persons:
                x1, y1, x2, y2 = person["box"]
                track_id = person.get("track_id")
                cv2.rectangle(evidence, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(evidence, f"PERSON ID:{track_id}", (x1, max(120, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

            for waste in wastes:
                x1, y1, x2, y2 = waste["box"]
                name = waste.get("class_name", "WASTE")
                conf = float(waste.get("confidence", 0.0))
                cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(evidence, f"{name} {conf:.2f}", (x1, max(120, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            filename = "illegal_dumping_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            path = os.path.join(self.cfg.output_dir, filename)
            ok = cv2.imwrite(path, evidence, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
            if not ok or not os.path.isfile(path) or os.path.getsize(path) == 0:
                print("Evidence image write failed:", path)
                return None

            self.last_evidence_url = f"/evidence/{filename}"
            print("Evidence saved and verified:", path)
            return self.last_evidence_url
        except Exception as exc:
            print("Evidence creation error:", exc)
            return None
