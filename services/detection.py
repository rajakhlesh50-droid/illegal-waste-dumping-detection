from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


Box = List[int]
Point = Tuple[int, int]
Detection = Dict[str, Any]


@dataclass
class DetectionConfig:
    waste_confidence: float = 0.20
    person_confidence: float = 0.45
    bottle_confidence: float = 0.25
    image_size: int = 512
    hand_waste_distance: float = 110.0
    false_waste_iou: float = 0.35
    false_waste_area_ratio: float = 0.20
    run_bottle_fallback: bool = True
    class_aliases: Dict[str, str] = field(default_factory=lambda: {
        "plastic_bottle": "BOTTLE",
        "plastic bottle": "BOTTLE",
        "bottle": "BOTTLE",
        "plastic": "PLASTIC",
        "paper": "PAPER",
        "cardboard": "CARDBOARD",
        "glass": "GLASS",
        "metal": "METAL",
        "can": "CAN",
        "tin": "CAN",
        "food": "FOOD WASTE",
        "food_waste": "FOOD WASTE",
        "organic": "ORGANIC WASTE",
        "organic_waste": "ORGANIC WASTE",
        "e_waste": "E-WASTE",
        "electronic": "E-WASTE",
        "waste": "WASTE",
    })


def normalize_class_name(name: Any, aliases: Dict[str, str]) -> str:
    raw = str(name).strip()
    key = raw.lower().replace("-", "_")
    return aliases.get(key, raw.upper())


def box_center(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def distance(a: Optional[Point], b: Optional[Point]) -> float:
    if a is None or b is None:
        return float("inf")
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


class DetectionService:
    """YOLO detection wrapper with no application-global state."""

    def __init__(self, waste_model, pose_model=None, bottle_model=None, config: Optional[DetectionConfig] = None):
        self.waste_model = waste_model
        self.pose_model = pose_model
        self.bottle_model = bottle_model
        self.cfg = config or DetectionConfig()

    def _extract_box(self, result, index: int) -> Box:
        xyxy = result.boxes.xyxy[index].cpu().numpy().tolist()
        return [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]

    def detect_persons(self, frame) -> List[Detection]:
        if self.pose_model is None:
            return []

        persons: List[Detection] = []
        try:
            results = self.pose_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=self.cfg.person_confidence,
                imgsz=self.cfg.image_size,
                verbose=False,
            )
        except Exception as exc:
            print("Person detection error:", exc)
            return persons

        if not results:
            return persons
        result = results[0]
        if result.boxes is None:
            return persons

        for i in range(len(result.boxes)):
            box = self._extract_box(result, i)
            confidence = float(result.boxes.conf[i].cpu().item())
            track_id = None
            if result.boxes.id is not None:
                try:
                    track_id = int(result.boxes.id[i].cpu().item())
                except Exception:
                    track_id = None

            wrists: List[Point] = []
            if result.keypoints is not None:
                try:
                    keypoints = result.keypoints.xy[i].cpu().numpy()
                    for kp_idx in (9, 10):
                        if len(keypoints) > kp_idx:
                            x, y = keypoints[kp_idx]
                            if x > 0 and y > 0:
                                wrists.append((int(x), int(y)))
                except Exception:
                    wrists = []

            persons.append({
                "box": box,
                "confidence": confidence,
                "track_id": track_id,
                "wrists": wrists,
            })

        return persons

    def detect_waste(self, frame) -> List[Detection]:
        wastes: List[Detection] = []

        try:
            results = self.waste_model.predict(
                frame,
                conf=self.cfg.waste_confidence,
                imgsz=self.cfg.image_size,
                verbose=False,
            )
            if results and results[0].boxes is not None:
                result = results[0]
                for i in range(len(result.boxes)):
                    box = self._extract_box(result, i)
                    conf = float(result.boxes.conf[i].cpu().item())
                    class_id = int(result.boxes.cls[i].cpu().item())
                    try:
                        raw_name = result.names[class_id]
                    except Exception:
                        raw_name = "waste"
                    class_name = normalize_class_name(raw_name, self.cfg.class_aliases)
                    wastes.append({
                        "box": box,
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": class_name,
                        "source": "custom_waste",
                    })
        except Exception as exc:
            print("Waste detection error:", exc)

        if self.cfg.run_bottle_fallback and self.bottle_model is not None:
            try:
                results = self.bottle_model.predict(
                    frame,
                    conf=self.cfg.bottle_confidence,
                    classes=[39],  # COCO bottle
                    imgsz=self.cfg.image_size,
                    verbose=False,
                )
                if results and results[0].boxes is not None:
                    result = results[0]
                    for i in range(len(result.boxes)):
                        box = self._extract_box(result, i)
                        conf = float(result.boxes.conf[i].cpu().item())
                        if any(iou(box, existing["box"]) >= 0.50 for existing in wastes):
                            continue
                        wastes.append({
                            "box": box,
                            "confidence": conf,
                            "class_id": 39,
                            "class_name": "BOTTLE",
                            "source": "bottle_model",
                        })
            except Exception as exc:
                print("Bottle detection error:", exc)

        return wastes

    def waste_near_hand(self, person: Detection, waste: Detection, threshold: Optional[float] = None) -> bool:
        wrists = person.get("wrists", [])
        if not wrists:
            return False
        target = box_center(waste["box"])
        limit = threshold if threshold is not None else self.cfg.hand_waste_distance
        return any(distance(wrist, target) <= limit for wrist in wrists)

    def find_holding_pair(self, persons: Iterable[Detection], wastes: Iterable[Detection]):
        for person in persons:
            for waste in wastes:
                if self.waste_near_hand(person, waste):
                    return person, waste
        return None, None

    def filter_false_waste_detections(self, wastes: List[Detection], persons: List[Detection]) -> List[Detection]:
        if not persons:
            return list(wastes)

        filtered: List[Detection] = []
        for waste in wastes:
            keep = True
            waste_box = waste["box"]
            waste_name = str(waste.get("class_name", "waste")).lower()
            waste_area = box_area(waste_box)

            for person in persons:
                person_box = person["box"]
                person_area = box_area(person_box)
                center = box_center(waste_box)
                inside = (
                    person_box[0] <= center[0] <= person_box[2]
                    and person_box[1] <= center[1] <= person_box[3]
                )
                large_relative = person_area > 0 and waste_area / person_area >= self.cfg.false_waste_area_ratio
                overlap = iou(waste_box, person_box)
                near_hand = self.waste_near_hand(person, waste, max(self.cfg.hand_waste_distance, 120.0))

                if waste_name in {"waste", "generic waste"} and not near_hand and (
                    overlap >= self.cfg.false_waste_iou or (inside and large_relative)
                ):
                    keep = False
                    break

            if keep:
                filtered.append(waste)

        return filtered
