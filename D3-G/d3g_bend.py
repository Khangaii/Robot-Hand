"""D3-G hand tracking. Preview: python d3g_bend.py; USB: add --port COM11 (actual port).

No calibration happens on the board. The open/fist calibration
(calibration.json) and the per-finger signal blend (finger_tuning.json) are
made on the laptop with the web UI (app.py) and copied here, beside this file.
"""
import argparse
from collections import deque
import math
from statistics import median
import time
from pathlib import Path

from camera import open_camera

import cv2
import mediapipe as mp
import numpy as np
import finger_tuning
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

HERE = Path(__file__).resolve().parent


def load_setup(calib_path, tuning_path):
    """Load laptop-made calibration + tuning; fail early with a clear reason."""
    cal_open, cal_closed = finger_tuning.load_calibration(calib_path)
    if cal_open is None:
        raise FileNotFoundError(
            f"No valid calibration at {calib_path}. Calibrate on the laptop "
            "(python app.py) and copy calibration.json here. Files from the "
            "old calibration format will not work; redo it with the new app.")
    if tuning_path.is_file():
        settings = finger_tuning.load_settings(tuning_path)
        print(f"Tuning: {tuning_path.name}")
    else:
        settings = finger_tuning.default_settings()
        print(f"Tuning: {tuning_path.name} not found - using built-in defaults "
              "(tuned thumb, original PIP-only for the other fingers).")
    bad = [n for n in NAMES
           if finger_tuning.compute(n, cal_closed[n], cal_open[n],
                                    cal_closed[n], settings[n])["output"] is None]
    if bad:
        raise RuntimeError("No usable signal for: " + ", ".join(bad) +
                           ". Recalibrate on the laptop or adjust the weights.")
    return cal_open, cal_closed, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Actual Arduino/ESP32 COM port; omit for camera preview")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration", type=Path, default=HERE / "calibration.json")
    parser.add_argument("--tuning", type=Path, default=HERE / "finger_tuning.json")
    args = parser.parse_args()
    cal_open, cal_closed, settings = load_setup(args.calibration, args.tuning)
    model = Path(__file__).resolve().with_name("hand_landmarker.task")
    if not model.is_file():
        alternative = model.with_name("hand_landmarker(1).task")
        if alternative.is_file():
            model = alternative
        else:
            raise FileNotFoundError("Place hand_landmarker.task beside this Python file.")
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model.read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
    )
    cap = open_camera(args.camera)
    link = None
    try:
        if args.port:
            import serial
            link = serial.Serial(args.port, 115200, timeout=0, write_timeout=0.3)
            time.sleep(2)  # USB opening may reset the board.
            link.reset_input_buffer()
        stabilizer = BendStabilizer()
        last_timestamp = -1
        last_sent = 0.0
        reply_buffer = ""
        status = "Show ONE hand (the calibrated hand)."
        print("Using saved calibration. ESC: quit.")
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("No frame from D3-G camera")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
                last_timestamp = timestamp
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                feats = measured = values = None
                h, w = frame.shape[:2]
                if result.hand_landmarks and result.hand_world_landmarks:
                    pixels = [(int(p.x * w), int(p.y * h))
                              for p in result.hand_landmarks[0]]
                    for a, b in CONNECTIONS:
                        cv2.line(frame, pixels[a], pixels[b], (0, 220, 0), 2)
                    for p in pixels:
                        cv2.circle(frame, p, 4, (0, 0, 255), -1)
                    try:
                        feats = finger_tuning.hand_features(
                            [(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]])
                    except ValueError:
                        feats = None
                if feats is not None:
                    measured = np.array([finger_tuning.compute(
                        n, feats[n], cal_open[n], cal_closed[n], settings[n])["output"]
                        for n in NAMES])
                    values = stabilizer.update(measured, time.monotonic())
                    status = "READY: sending commands." if link else "READY: preview only."
                else:
                    stabilizer.reset()
                    status = "NO HAND - output stopped. Show the same hand."
                now = time.monotonic()
                if link and now - last_sent >= 0.05:
                    packet = "STOP" if values is None else ",".join(map(str, values))
                    link.write((packet + "\n").encode("ascii"))
                    last_sent = now
                if link and link.in_waiting:
                    reply_buffer += link.read(min(link.in_waiting, 4096)).decode("ascii", errors="replace")
                    while "\n" in reply_buffer:
                        line, reply_buffer = reply_buffer.split("\n", 1)
                        print("Board:", line.strip())
                    reply_buffer = reply_buffer[-4096:]
                panel = np.zeros((310, w, 3), dtype=np.uint8)
                cv2.putText(panel, "Saved calibration | ESC: quit", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(panel, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 220, 255), 1)
                cv2.putText(panel, "Finger    Measured    Command", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                for i, name in enumerate(NAMES):
                    actual = "--" if measured is None else f"{measured[i]:.0f}%"
                    label = "--" if values is None else f"{values[i]}%"
                    held = ""
                    if values is not None:
                        if stabilizer.closed[i]:
                            held = " CLOSED"
                        elif stabilizer.middle[i]:
                            held = " MIDDLE"
                        elif stabilizer.opened[i]:
                            held = " OPEN"
                    cv2.putText(panel, f"{name:6}    {actual:>4}        {label:>4}{held}",
                                (10, 108 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 0), 1)
                cv2.putText(panel, "MIDDLE: 40-60% -> command 50% (all fingers).",
                            (10, 247), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                cv2.putText(panel, "OPEN: <=10% | CLOSED: >=65% (release <=60%).",
                            (10, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                cv2.putText(panel, "States confirmed after 0.12s; outside zones: tracking.",
                            (10, 297), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
                cv2.imshow("Robot Hand - D3-G", np.vstack((frame, panel)))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if link:
            try:
                link.write(b"STOP\n")
            finally:
                link.close()


if __name__ == "__main__":
    main()
