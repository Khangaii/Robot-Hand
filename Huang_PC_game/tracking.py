"""Hand-tracking helpers for huang_live.py (stabilizer, skeleton, finger tuning)."""
import argparse
import sys
from collections import deque
import math
from statistics import median
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
# finger_tuning.py may sit beside this file or in the folder above it.
for _dir in (HERE, HERE.parent):
    if (_dir / "finger_tuning.py").is_file():
        sys.path.insert(0, str(_dir))
        break
import finger_tuning  # noqa: E402

NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
          (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
CONNECTIONS = [(chain[j], chain[j + 1]) for chain in CHAINS for j in range(4)]
CONNECTIONS += [(5, 9), (9, 13), (13, 17)]

# 다섯 손가락에 동일하게 적용하며, 각 손가락의 상태는 독립적입니다.
OPEN_ENTER = 10.0
OPEN_RELEASE = 15.0
MIDDLE_MIN = 40.0
MIDDLE_MAX = 60.0
CLOSE_ENTER = 65.0
# 닫힘에서 내려올 때 60%에 도달하면 중간 단계로 이동할 수 있습니다.
CLOSE_RELEASE = MIDDLE_MAX
STATE_CONFIRM_SECONDS = 0.12
SMOOTH_SECONDS = 0.08
OUTPUT_DEADBAND = 3


class BendStabilizer:
    """필터링한 굽힘을 OPEN / MIDDLE / CLOSED / TRACKING으로 처리합니다."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = [deque(maxlen=3) for _ in NAMES]
        self.filtered = None
        self.output = None
        self.state = ["TRACKING"] * len(NAMES)
        self.pending_state = [None] * len(NAMES)
        self.pending_since = [None] * len(NAMES)
        self.opened = [False] * len(NAMES)
        self.middle = [False] * len(NAMES)
        self.closed = [False] * len(NAMES)
        self.last_time = None

    def update(self, measured, now):
        measured = [float(v) for v in measured]
        if len(measured) != len(NAMES) or not all(math.isfinite(v) for v in measured):
            raise ValueError("Invalid bend percentages")
        measured = [min(100.0, max(0.0, v)) for v in measured]
        for history, value in zip(self.history, measured):
            history.append(value)
        current = [median(history) for history in self.history]
        if self.filtered is None:
            self.filtered = current
        else:
            dt = max(0.0, now - self.last_time)
            alpha = 1.0 - math.exp(-dt / SMOOTH_SECONDS)
            self.filtered = [old + alpha * (new - old)
                             for old, new in zip(self.filtered, current)]
        self.last_time = now
        command = []
        for i, value in enumerate(self.filtered):
            state = self.state[i]
            # 필터가 경계에 한없이 접근할 때의 부동소수점 오차를 정리합니다.
            value = round(value, 2)
            # 닫힌 상태에서도 40~60%로 돌아오면 중간 단계로 전환합니다.
            if MIDDLE_MIN <= value <= MIDDLE_MAX:
                target = "MIDDLE"
            elif value >= CLOSE_ENTER or (state == "CLOSED" and value > CLOSE_RELEASE):
                target = "CLOSED"
            elif value <= OPEN_ENTER or (state == "OPEN" and value < OPEN_RELEASE):
                target = "OPEN"
            else:
                target = "TRACKING"

            if target == state:
                self.pending_state[i] = None
                self.pending_since[i] = None
            elif target != self.pending_state[i]:
                self.pending_state[i] = target
                self.pending_since[i] = now
            elif now - self.pending_since[i] >= STATE_CONFIRM_SECONDS:
                self.state[i] = target
                self.pending_state[i] = None
                self.pending_since[i] = None

            state = self.state[i]
            self.opened[i] = state == "OPEN"
            self.middle[i] = state == "MIDDLE"
            self.closed[i] = state == "CLOSED"
            if self.opened[i]:
                candidate = 0
            elif self.middle[i]:
                candidate = 50
            elif self.closed[i]:
                candidate = 100
            else:
                candidate = int(round(value))
                if self.output is not None:
                    previous = self.output[i]
                    if abs(candidate - previous) < OUTPUT_DEADBAND and candidate not in (0, 100):
                        candidate = previous
            command.append(candidate)
        self.output = command
        return command.copy()


# Legacy single-angle helpers. huang_live.py no longer uses them; kept in case
# something else imports them from this file.
def bend_scores(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Invalid landmarks")

    scores = []
    for finger_index, chain in enumerate(CHAINS):
        if finger_index == 0:
            # Thumb: MCP -> IP -> tip
            # Measure bending at the IP joint.
            chain = (2, 3, 4)
        else:
            chain = chain[1:4]

        segments = np.diff(points[list(chain)], axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths < 1e-8):
            raise ValueError("Degenerate landmarks")

        unit = segments / lengths[:, None]
        cosines = np.clip(
            np.sum(unit[:-1] * unit[1:], axis=1), -1, 1
        )
        scores.append(np.degrees(np.arccos(cosines)).mean())

    return np.array(scores)


def percentages(raw, opened, closed):
    span = np.asarray(closed) - np.asarray(opened)
    if not np.isfinite(span).all() or np.any(np.abs(span) < 12):
        raise ValueError("Open/closed calibration too similar")
    return np.clip((np.asarray(raw) - opened) / span * 100, 0, 100)


def load_tuning(path=None):
    """Per-finger settings from finger_tuning.json (beside this file, else the
    folder above it). Falls back to built-in defaults if there is none."""
    if path is None:
        path = next((d / "finger_tuning.json" for d in (HERE, HERE.parent)
                     if (d / "finger_tuning.json").is_file()), None)
    if path is not None and Path(path).is_file():
        print(f"Tuning: {path}")
        return finger_tuning.load_settings(Path(path))
    print("Tuning: finger_tuning.json not found - using built-in defaults "
          "(tuned thumb, original PIP-only for the other fingers).")
    return finger_tuning.default_settings()
