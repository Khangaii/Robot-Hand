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

import finger_tuning

# Reuse the existing, already-tuned stabilizer / skeleton drawing.
from camera_hand_mid50_v2 import (
    CONNECTIONS,
    NAMES,
    BendStabilizer,
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

        self.capture = None            # open | closed | finger_open | finger_closed
        self.capture_finger = None
        self.samples = []
        self.last_feats = None
        self.calib_file = HERE / "calibration.json"
        self.tuning_file = HERE / "finger_tuning.json"
        self.tuning = finger_tuning.load_settings(self.tuning_file)
        self.cal_open, self.cal_closed = finger_tuning.load_calibration(
            self.calib_file)

        self.state = {}
        self.fps = 0.0
        self._last_sent = 0.0
        self._reply_buffer = ""

    # ---- calibration
    def _calibrated(self):
        return self.cal_open is not None and self.cal_closed is not None

    def _save_calibration(self):
        finger_tuning.save_calibration(
            self.calib_file, self.cal_open, self.cal_closed)

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
            if not self._calibrated():
                return False, "Calibrate OPEN and FIST first."
            if not self.rps.start(time.monotonic()):
                return False, "A round is already in progress."
        return True, ""

    def start_calibration(self, pose, finger=None):
        with self.lock:
            if self.last_feats is None:
                return False, "Show your hand to the camera first."
            if pose == "open":
                self.cal_open = self.cal_closed = None
            elif pose == "closed":
                if self.cal_open is None:
                    return False, "Calibrate OPEN first."
                self.cal_closed = None
            elif pose in ("finger_open", "finger_closed"):
                if not self._calibrated():
                    return False, "Calibrate OPEN and FIST first."
                if finger not in finger_tuning.FINGER_NAMES:
                    return False, "Unknown finger"
                self.capture_finger = finger
            else:
                return False, "Unknown pose"
            self.capture = pose
            self.samples.clear()
            self.stabilizer.reset()
        return True, ""

    # ---- live finger tuning
    def update_tuning(self, finger, data, save=False):
        with self.lock:
            if finger not in self.tuning:
                return False, "Unknown finger"
            self.tuning[finger] = finger_tuning.sanitize_finger(
                finger, data, self.tuning[finger])
            if save:
                finger_tuning.save_settings(self.tuning_file, self.tuning)
        return True, ""

    def apply_preset(self, finger, preset_id):
        with self.lock:
            if finger not in self.tuning:
                return False, "Unknown finger"
            new = finger_tuning.apply_preset(
                finger, preset_id, self.tuning[finger])
            if new is None:
                return False, "Unknown preset"
            self.tuning[finger] = new
            finger_tuning.save_settings(self.tuning_file, self.tuning)
        return True, ""

    def reset_finger(self, finger):
        with self.lock:
            if finger not in self.tuning:
                return False, "Unknown finger"
            self.tuning[finger] = finger_tuning.original_finger(finger)
            finger_tuning.save_settings(self.tuning_file, self.tuning)
        return True, ""

    # ---- per-frame processing (called with lock held)
    def _process(self, feats, landmarks, now):
        self.last_feats = feats
        status = "Show ONE hand to the camera."
        measured = values = None

        if self.capture is not None:
            if feats is None:
                self.samples.clear()
            else:
                self.samples.append(finger_tuning.hand_vector(feats))
            label = self.capture.replace("finger_", self.capture_finger + " "
                                         if self.capture_finger else "")
            status = f"Hold {label.upper()} pose: {len(self.samples)}/30"
            if len(self.samples) >= 30:
                avg = finger_tuning.split_vector(
                    np.median(self.samples, axis=0))
                f = self.capture_finger
                if self.capture == "open":
                    self.cal_open = avg
                    status = ("OPEN saved. Now make a fist and press "
                              "Calibrate FIST.")
                elif self.capture == "closed":
                    self.cal_closed = avg
                    self._save_calibration()
                    status = "FIST saved."
                elif self.capture == "finger_open":
                    self.cal_open[f] = avg[f]
                    self._save_calibration()
                    status = f"{f} OPEN saved."
                else:
                    self.cal_closed[f] = avg[f]
                    self._save_calibration()
                    status = f"{f} FIST saved."
                self.capture = None
                self.samples.clear()

        calibrated = self._calibrated()
        tuning = {}
        for n in NAMES:
            tuning[n] = finger_tuning.compute(
                n,
                None if feats is None else feats[n],
                self.cal_open[n] if calibrated else None,
                self.cal_closed[n] if calibrated else None,
                self.tuning[n])

        if feats is not None and calibrated and self.capture is None:
            outputs = [tuning[n]["output"] for n in NAMES]
            if None in outputs:
                bad = [n for n, o in zip(NAMES, outputs) if o is None]
                status = ("No usable signal for " + ", ".join(bad) +
                          ". Raise a weight or recapture OPEN/FIST.")
                self.stabilizer.reset()
            else:
                measured = np.array(outputs)
                values = self.stabilizer.update(measured, now)
                status = "Ready."
        else:
            self.stabilizer.reset()
            if self.capture is None:
                if not calibrated:
                    status = "Calibrate OPEN, then FIST."
                elif feats is None:
                    status = "NO HAND - output stopped."

        # Decide what the robot should do.
        if self.mode == "follow":
            command = values              # None -> STOP
        else:
            gesture = classify(measured)
            command = self.rps.step(now, gesture) if calibrated else None

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
                "open": self.cal_open is not None,
                "closed": calibrated,
                "capturing": self.capture,
                "finger": self.capture_finger,
                "progress": len(self.samples),
            },
            "fingers": fingers,
            "tuning": {
                "fingers": {n: {
                    "settings": self.tuning[n],
                    "features": tuning[n]["features"],
                    "base": tuning[n]["base"],
                    "output": (None if tuning[n]["output"] is None
                               else round(tuning[n]["output"], 1)),
                } for n in NAMES},
                "presets": {
                    "thumb": finger_tuning.preset_list("Thumb"),
                    "finger": finger_tuning.preset_list("Index"),
                },
            },
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

                    feats = landmarks = None
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
                            feats = finger_tuning.hand_features(
                                [(p.x, p.y, p.z)
                                 for p in result.hand_world_landmarks[0]])
                        except ValueError:
                            feats = None

                    now = time.monotonic()
                    dt = now - prev
                    prev = now
                    if dt > 0:
                        self.fps = 0.9 * self.fps + 0.1 / dt

                    with self.lock:
                        command = self._process(feats, landmarks, now)
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
    body = request.get_json(silent=True) or {}
    return reply(engine.start_calibration(body.get("pose"),
                                          body.get("finger")))


@app.route("/api/rps/start", methods=["POST"])
def api_rps_start():
    return reply(engine.start_round())


@app.route("/api/tuning", methods=["POST"])
def api_tuning():
    body = request.get_json(silent=True) or {}
    return reply(engine.update_tuning(body.get("finger"),
                                      body.get("settings"),
                                      bool(body.get("save"))))


@app.route("/api/tuning/preset", methods=["POST"])
def api_tuning_preset():
    body = request.get_json(silent=True) or {}
    return reply(engine.apply_preset(body.get("finger"), body.get("id")))


@app.route("/api/tuning/reset", methods=["POST"])
def api_tuning_reset():
    return reply(engine.reset_finger(
        (request.get_json(silent=True) or {}).get("finger")))


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
