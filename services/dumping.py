from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from .detection import box_center, distance, DetectionService
except ImportError:
    from detection import box_center, distance, DetectionService


@dataclass
class DumpingConfig:
    required_holding_frames: int = 4
    required_release_frames: int = 3
    hand_waste_distance: float = 110.0
    release_distance: float = 65.0
    min_release_motion: float = 4.0
    fast_release_motion: float = 10.0
    dumping_event_cooldown: float = 30.0
    no_person_grace_frames: int = 4
    personless_warmup_frames: int = 20
    personless_confirm_frames: int = 3
    personless_min_motion: float = 25.0
    personless_min_confidence: float = 0.35


class DumpingEngine:
    """Stateful event engine for person-based and person-independent dumping."""

    def __init__(self, detection: DetectionService, config: Optional[DumpingConfig] = None):
        self.detection = detection
        self.cfg = config or DumpingConfig()
        self.reset()
        self.frame_index = 0
        self.last_personless_centers: Dict[str, tuple] = {}
        self.personless_streak = 0
        self.personless_baseline_ready = False
        self.recent_waste_keys: set[str] = set()
        self.last_event_time = 0.0

    def reset(self) -> None:
        self.holding_streak = 0
        self.release_streak = 0
        self.holding_confirmed = False
        self.release_confirmed = False
        self.holding_person_id = None
        self.holding_waste_center = None
        self.holding_waste_box = None
        self.holding_waste_class = "WASTE"
        self.holding_waste_confidence = 0.0
        self.last_waste_center = None
        self.last_wrist_center = None
        self.no_person_streak = 0
        self.current_interaction = False
        self.current_holding = False
        self.current_releasing = False
        self.current_illegal_dumping = False
        self.current_reason = ""
        self.current_release_progress = 0
        self.current_missing_progress = 0

    def _holding_pair(self, persons: List[dict], wastes: List[dict]):
        return self.detection.find_holding_pair(persons, wastes)

    def _confirm_personless(self, wastes: List[dict]) -> bool:
        if self.frame_index < self.cfg.personless_warmup_frames:
            return False
        if not wastes:
            self.personless_streak = 0
            return False

        # Use a stable coarse key per class. This is deliberately conservative:
        # a personless event requires a detected object to move enough frames in a row.
        moving_candidates = []
        for idx, waste in enumerate(wastes):
            conf = float(waste.get("confidence", 0.0))
            if conf < self.cfg.personless_min_confidence:
                continue
            center = box_center(waste["box"])
            key = f"{waste.get('class_name', 'WASTE')}:{idx}"
            prev = self.last_personless_centers.get(key)
            moved = prev is not None and distance(center, prev) >= self.cfg.personless_min_motion
            moving_candidates.append((key, center, moved))

        if any(moved for _, _, moved in moving_candidates):
            self.personless_streak += 1
        else:
            self.personless_streak = max(0, self.personless_streak - 1)

        self.last_personless_centers = {key: center for key, center, _ in moving_candidates}
        return self.personless_streak >= self.cfg.personless_confirm_frames

    def process(self, persons: List[dict], wastes: List[dict], frame=None) -> Dict[str, Any]:
        self.frame_index += 1
        self.current_interaction = False
        self.current_holding = False
        self.current_releasing = False
        self.current_illegal_dumping = False
        self.current_reason = ""
        self.current_release_progress = self.release_streak
        self.current_missing_progress = 0

        # Person-independent mode: ONLY for a newly/moving detected waste.
        # This does not require a person or a prior holding state.
        personless_confirmed = self._confirm_personless(wastes) if not persons and not self.holding_confirmed else False
        if personless_confirmed and time.time() - self.last_event_time >= self.cfg.dumping_event_cooldown:
            self.current_illegal_dumping = True
            self.current_reason = "PERSON_NOT_REQUIRED_NEW_OR_MOVING_WASTE"
            self.last_event_time = time.time()
            self.reset()
            return self.result(personless=True, confirmed=True, evidence_wastes=wastes[-3:])

        # Person-based state machine.
        person, waste = self._holding_pair(persons, wastes) if wastes else (None, None)
        if person is not None and waste is not None:
            self.current_interaction = True
            self.current_holding = True
            self.release_streak = 0
            self.release_confirmed = False
            center = box_center(waste["box"])
            self.holding_person_id = person.get("track_id")

            if not self.holding_confirmed:
                self.holding_streak += 1
                if self.holding_streak >= self.cfg.required_holding_frames:
                    self.holding_confirmed = True
                    self.holding_waste_center = center
                    self.holding_waste_box = list(waste["box"])
                    self.holding_waste_class = waste.get("class_name", "WASTE")
                    self.holding_waste_confidence = float(waste.get("confidence", 0.0))
                    wrists = person.get("wrists", [])
                    self.last_wrist_center = min(wrists, key=lambda p: distance(p, center)) if wrists else None
                    self.last_waste_center = center
                    print("WASTE HOLDING CONFIRMED")
            else:
                self.holding_waste_center = center
                self.holding_waste_box = list(waste["box"])
                self.holding_waste_class = waste.get("class_name", self.holding_waste_class)
                self.holding_waste_confidence = float(waste.get("confidence", self.holding_waste_confidence))
                wrists = person.get("wrists", [])
                self.last_wrist_center = min(wrists, key=lambda p: distance(p, center)) if wrists else self.last_wrist_center
                self.last_waste_center = center
            return self.result()

        if not self.holding_confirmed:
            self.holding_streak = 0
            return self.result()

        # Confirmed hold but waste is no longer close to the hand.
        selected = None
        for candidate in persons:
            if candidate.get("track_id") == self.holding_person_id:
                selected = candidate
                break

        if selected is None:
            self.no_person_streak += 1
        else:
            self.no_person_streak = 0

        nearest_waste = None
        if wastes:
            reference = self.holding_waste_center or self.last_waste_center
            if reference is None and selected is not None:
                reference = box_center(selected["box"])
            nearest_waste = min(wastes, key=lambda w: distance(box_center(w["box"]), reference)) if reference else wastes[0]

        if nearest_waste is None:
            self.release_streak += 1
            self.current_missing_progress = self.release_streak
            self.current_releasing = True
        else:
            center = box_center(nearest_waste["box"])
            motion = distance(center, self.last_waste_center)
            wrists = selected.get("wrists", []) if selected else []
            wrist_distance = min((distance(w, center) for w in wrists), default=float("inf"))
            far_from_hand = wrist_distance >= self.cfg.release_distance
            moving_away = motion >= self.cfg.min_release_motion
            fast_release = motion >= self.cfg.fast_release_motion
            if (moving_away and far_from_hand) or (fast_release and far_from_hand) or (selected is None and moving_away):
                self.release_streak += 1
                self.current_releasing = True
            else:
                self.release_streak = max(0, self.release_streak - 1)
                self.current_releasing = self.release_streak > 0
            self.last_waste_center = center

        self.current_release_progress = self.release_streak
        if self.release_streak >= self.cfg.required_release_frames:
            self.release_confirmed = True

        if self.release_confirmed and time.time() - self.last_event_time >= self.cfg.dumping_event_cooldown:
            self.current_illegal_dumping = True
            self.current_reason = "WASTE_RELEASED_AFTER_HOLDING"
            self.last_event_time = time.time()
            evidence_wastes = [nearest_waste] if nearest_waste is not None else [{
                "box": self.holding_waste_box or [0, 0, 30, 30],
                "confidence": self.holding_waste_confidence,
                "class_name": self.holding_waste_class,
            }]
            self.reset()
            return self.result(confirmed=True, evidence_wastes=evidence_wastes)

        return self.result()

    def result(self, personless: bool = False, confirmed: bool = False, evidence_wastes: Optional[List[dict]] = None) -> Dict[str, Any]:
        return {
            "interaction": self.current_interaction,
            "holding": self.current_holding,
            "releasing": self.current_releasing,
            "illegal_dumping": self.current_illegal_dumping or confirmed,
            "confirmed": confirmed or self.current_illegal_dumping,
            "personless": personless,
            "reason": self.current_reason,
            "release_progress": self.current_release_progress,
            "missing_progress": self.current_missing_progress,
            "evidence_wastes": evidence_wastes or [],
            "person_id": self.holding_person_id,
        }
