"""D3-G rock-paper-scissors. Run beside d3g_bend.py in its working environment."""
import argparse
from collections import Counter
import math
from pathlib import Path
import random
import time

# Thumb, index, middle, ring, pinky. Percentages use the UNO's calibrated travel.
POSES = {
    "ROCK": (100, 100, 100, 100, 100),
    "PAPER": (0, 0, 0, 0, 0),
    "SCISSORS": (100, 0, 0, 100, 100),
}
BEATS = {"ROCK": "SCISSORS", "PAPER": "ROCK", "SCISSORS": "PAPER"}
PATTERNS = {tuple(v[1:]): k for k, v in POSES.items()}


def classify(values):
    if values is None:
        return None
    pattern = tuple(0 if v <= 30 else 100 if v >= 65 else None for v in values)
    return PATTERNS.get(pattern)


def verdict(player, robot):
    if player == robot:
        return "DRAW"
    return "YOU WIN" if BEATS[player] == robot else "ROBOT WINS"


def consensus(samples):
    if not samples:
        return None
    label, count = Counter(samples).most_common(1)[0]
    if label and count >= 2 and count / len(samples) >= 2 / 3:
        return label
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", help="UNO serial port (omit for preview)")
    parser.add_argument("--window", type=float, default=0.5, help="capture seconds")
    args = parser.parse_args()
    if not math.isfinite(args.window) or args.window <= 0:
        parser.error("--window must be a positive finite number")

    import d3g_bend as hand
    import cv2
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    folder = Path(hand.__file__).resolve().parent
    model = next((folder / name for name in ("hand_landmarker.task", "hand_landmarker(1).task")
                  if (folder / name).is_file()), None)
    if model is None:
        parser.error("Place hand_landmarker.task beside d3g_bend.py")
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model.read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    camera = hand.open_camera(args.camera)
    link = None
    try:
        if args.port:
            import serial
            link = serial.Serial(args.port, 115200, timeout=0, write_timeout=0.3)
            time.sleep(2)
            link.reset_input_buffer()

        state = "READY"
        opened = closed = calibration = robot = None
        samples, votes = [], []
        start = cue = last_send = 0.0
        last_stamp = -1
        message = "Show an open hand and press O."
        result = ""
        title = "Robot Hand | Rock Paper Scissors"
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                acquired = time.monotonic()
                if not ok or frame is None:
                    raise RuntimeError("No camera frame. Close the basic-mode program first.")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stamp = max(last_stamp + 1, int(acquired * 1000))
                last_stamp = stamp
                detection = detector.detect_for_video(
                    hand.mp.Image(image_format=hand.mp.ImageFormat.SRGB, data=rgb), stamp)
                raw = values = None
                if detection.hand_world_landmarks:
                    try:
                        raw = hand.bend_scores([(p.x, p.y, p.z)
                                                for p in detection.hand_world_landmarks[0]])[1:]
                    except ValueError:
                        pass
                if calibration:
                    if raw is None:
                        samples.clear()
                    else:
                        samples.append(raw)
                    message = f"Hold {calibration}: {len(samples)}/30"
                    if len(samples) == 30:
                        average = np.median(samples, axis=0)
                        if calibration == "OPEN":
                            opened = average
                            message = "Open saved. Make a fist and press C."
                        elif np.all(average - opened >= 12):
                            closed = average
                            message = "Calibration ready."
                        else:
                            message = "Small bend range. Make a tighter fist and press C."
                        calibration = None
                        samples.clear()
                if raw is not None and closed is not None:
                    values = hand.percentages(raw, opened, closed)
                player = classify(values)
                now = time.monotonic()
                if state == "COUNTDOWN" and now - start >= 3:
                    state = "SHOOT"
                    robot = random.choice(tuple(POSES))
                    votes.clear()
                if state == "CAPTURE":
                    if cue <= acquired <= cue + args.window:
                        votes.append(player)
                    if now >= cue + args.window:
                        chosen = consensus(votes)
                        result = (f"{chosen} vs {robot}  |  {verdict(chosen, robot)}"
                                  if chosen else "No clear throw. Try again.")
                        state = "RESULT"
                if state == "COUNTDOWN":
                    headline = str(max(1, math.ceil(3 - (now - start))))
                elif state in ("SHOOT", "CAPTURE"):
                    headline = "SHOW!"
                elif state == "RESULT":
                    headline = result
                else:
                    headline = "READY"

                # A fixed-width panel keeps text legible even with a small camera frame.
                width = max(640, frame.shape[1])
                canvas = np.zeros((frame.shape[0] + 166, width, 3), dtype=np.uint8)
                canvas[:frame.shape[0], :frame.shape[1]] = frame
                status = message
                if state == "READY" and closed is not None and not calibration:
                    status = "Live: " + (player or ("UNCLEAR" if raw is not None else "NO HAND"))
                    if values is not None:
                        status += "   I M R P: " + " ".join(str(round(v)) for v in values)
                rows = [(headline, 0.88), (status if state == "READY" else "", 0.55),
                        ("O: open   C: fist   SPACE: play again/start   ESC: quit", 0.55),
                        ("Ready/test keys: 1 rock   2 paper   3 scissors", 0.55),
                        ("USB: " + (args.port or "preview only"), 0.5)]
                for i, (line, scale) in enumerate(rows):
                    cv2.putText(canvas, line, (10, frame.shape[0] + 30 + 30*i),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (100, 255, 200), 1, cv2.LINE_AA)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(1) & 255
                if key == 27:
                    break
                if state == "SHOOT":
                    cue = time.monotonic()
                    last_send = 0.0
                    state = "CAPTURE"
                if state in ("READY", "RESULT"):
                    if key == 32 and closed is not None and calibration is None:
                        start = time.monotonic()
                        robot = None
                        state = "COUNTDOWN"
                    elif key in (ord("o"), ord("O"), ord("c"), ord("C")):
                        state = "READY"
                        robot = None
                        if chr(key).upper() == "O":
                            opened = closed = None
                            calibration = "OPEN"
                        elif opened is not None:
                            closed = None
                            calibration = "FIST"
                        samples.clear()
                    elif key in map(ord, "123"):
                        state = "READY"
                        robot = ("ROCK", "PAPER", "SCISSORS")[key - ord("1")]
                        message = "Test: " + robot
                        print(message, flush=True)
                now = time.monotonic()
                if link and now - last_send >= 0.05:
                    packet = ",".join(map(str, POSES[robot])) if robot else "STOP"
                    link.write((packet + "\n").encode("ascii"))
                    last_send = now
                    if link.in_waiting:
                        link.read(min(link.in_waiting, 4096))
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if link is not None:
            try:
                link.write(b"STOP\n")
            finally:
                link.close()


if __name__ == "__main__":
    main()
