"""
Robot hand control UI (Flask).

Run beside hand_detect.py and hand_landmarker.task:

    python app.py                       # laptop webcam, no robot
    python app.py --port COM5           # laptop webcam + Arduino
    python app.py --source http://...   # network camera / stream URL
    python app.py --host 0.0.0.0        # reachable from other machines

Then open http://localhost:5000
"""
import argparse
import json
import math
import platform
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Reuse the existing, already-tuned finger logic.
from hand_detect import (
    CONNECTIONS,
    NAMES,
    BendStabilizer,
    bend_scores,
    percentages,
)

HERE = Path(__file__).resolve().parent

LANDMARK_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip",
]

# ---------------------------------------------------------------- RPS config
COUNTDOWN_SECONDS = 3      # 3 -> 2 -> 1 -> reveal
CAPTURE_DELAY = 0.30       # wait after reveal so the throw has settled...
CAPTURE_SECONDS = 0.70     # ...then vote over this window
MIN_VALID_FRAMES = 4       # need at least this many readable frames
MIN_AGREEMENT = 0.60       # share of frames that must agree
EXTENDED_BELOW = 45.0      # finger bend % under this counts as "extended"

# Robot finger commands: 0 = open, 100 = closed (order: thumb..pinky)
POSES = {
    "rock":     [100, 100, 100, 100, 100],
    "paper":    [0, 0, 0, 0, 0],
    "scissors": [100, 0, 0, 100, 100],
}
READY_POSE = POSES["rock"]
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}


def classify(measured):
    """Return rock / paper / scissors / None from bend percentages.
    The thumb is ignored: it is the least reliable finger to read."""
    if measured is None:
        return None
    idx, mid, ring, pinky = [m < EXTENDED_BELOW for m in measured[1:]]
    if not (idx or mid or ring or pinky):
        return "rock"
    if idx and mid and ring and pinky:
        return "paper"
    if idx and mid and not ring and not pinky:
        return "scissors"
    return None


class RockPaperScissors:
    def __init__(self):
        self.score = {"win": 0, "lose": 0, "draw": 0}
        self.reset()

    def reset(self):
        self.phase = "idle"   # idle -> countdown -> capture -> result
        self.t_start = self.t_reveal = None
        self.robot = self.player = self.result = None
        self.votes = []
        self.message = "Press Start to play."

    def start(self, now):
        if self.phase in ("countdown", "capture"):
            return False
        self.reset()
        self.phase = "countdown"
        self.t_start = now
        self.message = "Get ready..."
        return True

    def step(self, now, gesture):
        """Advance the round; return the finger command for the robot."""
        if self.phase == "countdown":
            if now - self.t_start >= COUNTDOWN_SECONDS:
                self.robot = random.choice(list(POSES))
                self.t_reveal = now
                self.phase = "capture"
                self.message = "SHOOT!"

        elif self.phase == "capture":
            begin = self.t_reveal + CAPTURE_DELAY
            if begin <= now <= begin + CAPTURE_SECONDS:
                self.votes.append(gesture)
            elif now > begin + CAPTURE_SECONDS:
                self._resolve()

        if self.phase in ("capture", "result") and self.robot:
            return list(POSES[self.robot])
        return list(READY_POSE)

    def _resolve(self):
        valid = [v for v in self.votes if v]
        self.phase = "result"
        if len(valid) >= MIN_VALID_FRAMES:
            move, count = Counter(valid).most_common(1)[0]
            if count / len(valid) >= MIN_AGREEMENT:
                self.player = move
                if move == self.robot:
                    self.result = "draw"
                elif BEATS[move] == self.robot:
                    self.result = "win"
                else:
                    self.result = "lose"
                self.score[self.result] += 1
                self.message = {
                    "win": "You win!", "lose": "You lose!", "draw": "Draw!",
                }[self.result]
                return
        self.result = "void"
        self.message = "Couldn't read your throw. Round not counted."

    def snapshot(self, now):
        number = None
        if self.phase == "countdown":
            remaining = COUNTDOWN_SECONDS - (now - self.t_start)
            number = max(1, math.ceil(remaining))
        show_robot = self.phase in ("capture", "result")
        return {
            "phase": self.phase,
            "countdown": number,
            "robot_move": self.robot if show_robot else None,
            "player_move": self.player,
            "result": self.result,
            "message": self.message,
            "score": dict(self.score),
        }


# -------------------------------------------------------------------- engine
def open_capture(source):
    if isinstance(source, int) and platform.system() == "Windows":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera/source: {source!r}")
    return cap


def find_model():
    for name in ("hand_landmarker.task", "hand_landmarker(1).task"):
        path = HERE / name
        if path.is_file():
            return path
    raise FileNotFoundError("Place hand_landmarker.task beside app.py.")


class Engine:
    def __init__(self, source, port=None):
        self.cap = open_capture(source)
        self.model = find_model()

        self.link = None
        if port:
            import serial
            self.link = serial.Serial(port, 115200, timeout=0,
                                      write_timeout=0.3)
            time.sleep(2)  # UNO may reset on connect
            self.link.reset_input_buffer()

        self.lock = threading.RLock()
        self.frame_cond = threading.Condition()
        self.jpeg = None
        self.frame_id = 0
        self.stop_event = threading.Event()

        self.mode = "follow"          # "follow" | "rps"
        self.rps = RockPaperScissors()
        self.stabilizer = BendStabilizer()

        self.opened = self.closed = None
        self.capture = None
        self.samples = []
        self.last_raw = None
        self.calib_file = HERE / "calibration.json"
        self._load_calibration()

        self.state = {}
        self.fps = 0.0
        self._last_sent = 0.0
        self._reply_buffer = ""

    # ---- calibration persistence
    def _load_calibration(self):
        try:
            data = json.loads(self.calib_file.read_text())
            self.opened = np.array(data["open"], dtype=float)
            self.closed = np.array(data["closed"], dtype=float)
        except Exception:
            self.opened = self.closed = None

    def _save_calibration(self):
        try:
            self.calib_file.write_text(json.dumps({
                "open": self.opened.tolist(),
                "closed": self.closed.tolist(),
            }))
        except Exception as e:
            print("Could not save calibration:", e)

    # ---- commands from the web UI
    def set_mode(self, mode):
        if mode not in ("follow", "rps"):
            return False, "Unknown mode"
        with self.lock:
            if mode != self.mode:
                self.mode = mode
                self.rps.reset()
                self.stabilizer.reset()
        return True, ""

    def start_round(self):
        with self.lock:
            if self.mode != "rps":
                return False, "Switch to Rock Paper Scissors mode first."
            if self.opened is None or self.closed is None:
                return False, "Calibrate OPEN and FIST first."
            if not self.rps.start(time.monotonic()):
                return False, "A round is already in progress."
        return True, ""

    def start_calibration(self, pose):
        with self.lock:
            if self.last_raw is None:
                return False, "Show your hand to the camera first."
            if pose == "open":
                self.capture = "open"
                self.opened = self.closed = None
            elif pose == "closed":
                if self.opened is None:
                    return False, "Calibrate OPEN first."
                self.capture = "closed"
                self.closed = None
            else:
                return False, "Unknown pose"
            self.samples.clear()
            self.stabilizer.reset()
        return True, ""

    # ---- per-frame processing (called with lock held)
    def _process(self, raw, landmarks, now):
        self.last_raw = raw
        status = "Show ONE hand to the camera."
        measured = values = None

        if self.capture is not None:
            if raw is None:
                self.samples.clear()
            else:
                self.samples.append(raw.copy())
            status = f"Hold {self.capture.upper()} pose: {len(self.samples)}/30"
            if len(self.samples) >= 30:
                average = np.median(self.samples, axis=0)
                if self.capture == "open":
                    self.opened = average
                    status = "OPEN saved. Now make a fist and press Calibrate FIST."
                else:
                    self.closed = average
                    self._save_calibration()
                    status = "FIST saved."
                self.capture = None
                self.samples.clear()

        calibrated = self.opened is not None and self.closed is not None

        if raw is not None and calibrated and self.capture is None:
            try:
                measured = percentages(raw, self.opened, self.closed)
                values = self.stabilizer.update(measured, now)
                status = "Ready."
            except ValueError:
                bad = [NAMES[i] for i in range(5)
                       if abs(self.closed[i] - self.opened[i]) < 12]
                status = "Recalibrate: too little change in " + ", ".join(bad)
                self.stabilizer.reset()
        else:
            self.stabilizer.reset()
            if self.capture is None:
                if not calibrated:
                    status = "Calibrate OPEN, then FIST."
                elif raw is None:
                    status = "NO HAND - output stopped."

        # Decide what the robot should do.
        if self.mode == "follow":
            command = values              # None -> STOP
        else:
            gesture = classify(measured)
            if calibrated:
                command = self.rps.step(now, gesture)
            else:
                command = None

        m_list = None if measured is None else [float(v) for v in measured]
        fingers = []
        for i, name in enumerate(NAMES):
            st = "-"
            if self.mode == "follow" and values is not None:
                st = self.stabilizer.state[i]
            fingers.append({
                "name": name,
                "measured": None if m_list is None else round(m_list[i]),
                "command": None if command is None else command[i],
                "state": st,
            })

        self.state = {
            "mode": self.mode,
            "status": status,
            "hand_detected": landmarks is not None,
            "fps": round(self.fps, 1),
            "serial": self.link is not None,
            "calibration": {
                "open": self.opened is not None,
                "closed": self.closed is not None,
                "capturing": self.capture,
                "progress": len(self.samples),
            },
            "fingers": fingers,
            "landmarks": None if landmarks is None else [
                [round(v, 3) for v in p] for p in landmarks
            ],
            "live_gesture": classify(m_list) if m_list else None,
            "rps": self.rps.snapshot(now),
        }
        return command

    # ---- robot serial link
    def _send(self, command, now):
        if not self.link:
            return
        if now - self._last_sent >= 0.05:
            packet = "STOP" if command is None else ",".join(map(str, command))
            self.link.write((packet + "\n").encode("ascii"))
            self._last_sent = now
        if self.link.in_waiting:
            self._reply_buffer += self.link.read(
                min(self.link.in_waiting, 4096)
            ).decode("ascii", errors="replace")
            while "\n" in self._reply_buffer:
                line, self._reply_buffer = self._reply_buffer.split("\n", 1)
                print("Board:", line.strip())
            self._reply_buffer = self._reply_buffer[-4096:]

    # ---- main loop
    def run(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_buffer=self.model.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        last_ts = -1
        prev = time.monotonic()

        try:
            with vision.HandLandmarker.create_from_options(options) as det:
                while not self.stop_event.is_set():
                    ok, frame = self.cap.read()
                    if not ok:
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ts = max(time.monotonic_ns() // 1_000_000, last_ts + 1)
                    last_ts = ts
                    result = det.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                        ts)

                    raw = landmarks = None
                    h, w = frame.shape[:2]
                    if result.hand_landmarks and result.hand_world_landmarks:
                        pts = result.hand_landmarks[0]
                        landmarks = [(p.x, p.y, p.z) for p in pts]
                        pixels = [(int(p.x * w), int(p.y * h)) for p in pts]
                        for a, b in CONNECTIONS:
                            cv2.line(frame, pixels[a], pixels[b],
                                     (0, 220, 0), 2)
                        for p in pixels:
                            cv2.circle(frame, p, 4, (0, 0, 255), -1)
                        try:
                            raw = bend_scores([
                                (p.x, p.y, p.z)
                                for p in result.hand_world_landmarks[0]])
                        except ValueError:
                            raw = None

                    now = time.monotonic()
                    dt = now - prev
                    prev = now
                    if dt > 0:
                        self.fps = 0.9 * self.fps + 0.1 / dt

                    with self.lock:
                        command = self._process(raw, landmarks, now)
                    self._send(command, now)

                    ok, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self.frame_cond:
                            self.jpeg = buf.tobytes()
                            self.frame_id += 1
                            self.frame_cond.notify_all()
        finally:
            self.cap.release()
            if self.link:
                try:
                    self.link.write(b"STOP\n")
                finally:
                    self.link.close()

    def wait_frame(self, last_id, timeout=1.0):
        with self.frame_cond:
            self.frame_cond.wait_for(
                lambda: self.frame_id != last_id, timeout=timeout)
            return self.frame_id, self.jpeg

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            state["landmark_names"] = LANDMARK_NAMES
            return state


# ----------------------------------------------------------------- web app
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")
engine = None  # set in main()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/video")
def video():
    def generate():
        last = -1
        while True:
            last, jpg = engine.wait_frame(last)
            if jpg is None:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg + b"\r\n")
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(engine.snapshot())


def reply(result):
    ok, error = result
    return jsonify({"ok": ok, "error": error}), (200 if ok else 400)


@app.route("/api/mode", methods=["POST"])
def api_mode():
    return reply(engine.set_mode((request.get_json(silent=True) or {})
                                 .get("mode")))


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    return reply(engine.start_calibration(
        (request.get_json(silent=True) or {}).get("pose")))


@app.route("/api/rps/start", methods=["POST"])
def api_rps_start():
    return reply(engine.start_round())


def main():
    global engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Arduino serial port; omit for no robot")
    parser.add_argument("--source", default="0",
                        help="camera index or stream URL (default 0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=5000)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    engine = Engine(source, args.port)
    worker = threading.Thread(target=engine.run, daemon=True)
    worker.start()

    try:
        app.run(host=args.host, port=args.web_port,
                threaded=True, debug=False)
    finally:
        engine.stop_event.set()
        worker.join(timeout=3)


if __name__ == "__main__":
    sys.exit(main())
