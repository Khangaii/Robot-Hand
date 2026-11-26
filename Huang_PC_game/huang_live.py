"""Run beside the working d3g_bend.py in its existing Python environment.

Finger tracking uses finger_tuning.py / finger_tuning.json (see tracking.py).
Calibration is still the whole-hand O (open) and C (fist) only.
"""
import argparse
from pathlib import Path
import time
import re
import subprocess
from huang_input import ModeSwitch

from tracking import (BendStabilizer, CONNECTIONS, NAMES, cv2, finger_tuning,
                      load_tuning, mp, np, python, vision)
from huang_ui import FollowScreen, instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Actual Arduino/ESP32 COM port; omit for camera preview")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--switch", nargs=2, metavar=("GPIOCHIP", "OFFSET"))
    parser.add_argument("--board", action="store_true", help="D3-G camera and Windows game host")
    parser.add_argument("--tuning", type=Path, default=None,
                        help="finger_tuning.json (default: beside tracking.py, else the folder above)")
    args = parser.parse_args()
    if args.switch:
        try:
            offset = int(args.switch[1])
            if offset < 0:
                raise ValueError
        except ValueError:
            parser.error("OFFSET must be a nonnegative GPIO line offset")
    settings = load_tuning(args.tuning)
    if args.board:
        from camera import open_camera
        from game_link import Client
        game = Client()
        info = subprocess.check_output(["xrandr", "--current"], text=True)
        match = re.search(r"current (\d+) x (\d+)", info)
        desktop = tuple(map(int, match.groups())) if match else (960, 540)
    else:
        import ctypes
        from pc_game import Engine
        game = Engine()
        ctypes.windll.user32.SetProcessDPIAware()
        desktop = tuple(ctypes.windll.user32.GetSystemMetrics(i) for i in (0, 1))
    display_size = desktop
    print("VNC desktop:", desktop, flush=True)
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
    screen = FollowScreen()
    title = "SIOR | Robot Hand"
    fullscreen = True
    mode = "follow"
    selected = [mode]
    action = None
    shown = None

    def click(event, x, y, flags, userdata):
        x = x * 960 // display_size[0]
        y = y * 540 // display_size[1]
        if event == cv2.EVENT_LBUTTONUP and 14 <= y <= 54:
            if 678 <= x <= 803:
                selected[0] = "follow"
            elif 813 <= x <= 938:
                selected[0] = "game"

    if args.board:
        cap = open_camera(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Cannot open webcam. Close other camera apps, check permissions or try --camera 1.")
    link = switch = None
    try:
        if args.switch:
            switch = ModeSwitch(args.switch[0], offset)
        if args.port:
            import serial
            link = serial.Serial(args.port, 115200, timeout=0, write_timeout=0.3)
            time.sleep(2)  # USB opening may reset the board.
            link.reset_input_buffer()
        cal_open = cal_closed = None
        stabilizer = BendStabilizer()
        capture = None
        samples = []
        last_timestamp = -1
        last_sent = 0.0
        reply_buffer = ""
        print("Use the SAME hand. O: open calibration; C: fist calibration; ESC: quit.")
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(title, *desktop)
        cv2.moveWindow(title, 0, 0)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(title, click)
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                acquired = time.monotonic()
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
                if capture is not None:
                    if feats is None:
                        samples.clear()  # Require consecutive visible-hand frames.
                    else:
                        samples.append(finger_tuning.hand_vector(feats))
                    if len(samples) == 30:
                        average = finger_tuning.split_vector(np.median(samples, axis=0))
                        if capture == "open":
                            cal_open = average
                        else:
                            cal_closed = average
                        capture = None
                        samples.clear()
                if (feats is not None and cal_open is not None and cal_closed is not None
                        and capture is None):
                    outputs = [finger_tuning.compute(
                        n, feats[n], cal_open[n], cal_closed[n], settings[n])["output"]
                        for n in NAMES]
                    if None in outputs:  # no usable signal: the "redo" screen
                        stabilizer.reset()
                    else:
                        measured = np.array(outputs)
                        values = stabilizer.update(measured, time.monotonic())
                else:
                    stabilizer.reset()
                now = time.monotonic()
                if switch:
                    changed = switch.change(now)
                    if changed:
                        selected[0] = changed
                if mode != selected[0]:
                    mode = selected[0]
                    game.close()
                    action = shown = None
                    last_sent = 0.0
                output = values
                if mode == "game":
                    fingers = None if measured is None else measured.tolist()  # thumb..pinky
                    reply = game.frame([now, acquired, fingers, action, shown])
                    action = shown = None
                    state = reply["state"] if reply else "offline"
                    lines = reply["lines"] if reply else ["Waiting for PC", "Start pc_game.py"]
                    output = reply["output"] if reply else None
                    if state == "show":
                        last_sent = 0.0
                if link and now - last_sent >= 0.05:
                    packet = "STOP" if output is None else ",".join(map(str, output))
                    link.write((packet + "\n").encode("ascii"))
                    last_sent = now
                if link and link.in_waiting:
                    reply_buffer += link.read(min(link.in_waiting, 4096)).decode("ascii", errors="replace")
                    while "\n" in reply_buffer:
                        line, reply_buffer = reply_buffer.split("\n", 1)
                        print("Board:", line.strip())
                    reply_buffer = reply_buffer[-4096:]
                hint = instruction(feats, cal_open, cal_closed, capture, measured)
                if mode == "game" and hint == "ready":
                    hint = {"ready": "game_ready", "result": "game_result"}.get(state, "game_throw")
                canvas = screen.render(frame, values, hint, len(samples), stabilizer.state, mode)
                if mode == "game":
                    positions = [210 + i * 53 for i in range(len(lines))]
                    if state == "countdown" and lines[0] in ("1", "2", "3"):
                        canvas[196:331, 646:924] = screen.art["count_" + lines[0]]
                        lines = []
                    elif state == "result" and len(lines) == 4 and lines[2] in screen.art:
                        canvas[148:283, 646:924] = screen.art[lines[2]]
                        lines = [lines[0], lines[1], lines[3]]
                        positions = [318, 354, 415]
                    for y, line in zip(positions, lines):
                        cv2.putText(canvas, line, (654, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (53, 60, 23), 2, cv2.LINE_AA)
                display_size = desktop if fullscreen else (960, 540)
                if display_size != (960, 540):
                    canvas = cv2.resize(canvas, display_size, interpolation=cv2.INTER_LINEAR)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(1) & 0xFF
                if mode == "game":
                    shown = time.monotonic() if state == "show" else None
                    if (key == 32 and mode == selected[0] and values is not None
                            and state in ("ready", "result")):
                        action = "start"
                    if key in map(ord, "oOcC") and feats is not None:
                        action = "reset"
                if key in (ord("f"), ord("F")):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.resizeWindow(title, *desktop)
                    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:
                        cv2.resizeWindow(title, 960, 540)
                if key == 27:
                    break

                if key in (ord("o"), ord("O"), ord("c"), ord("C")):
                    if feats is None:
                        print("Show the same hand before calibration.")
                        continue
                    if key in (ord("o"), ord("O")):
                        capture = "open"
                        cal_open = cal_closed = None  # Begin a fresh calibration pair.
                    elif cal_open is not None:
                        capture = "closed"
                        cal_closed = None
                    else:
                        print("Press O with an open hand FIRST.")
                        continue
                    samples.clear()
                    stabilizer.reset()
    finally:
        game.close()
        if switch:
            switch.close()
        cap.release()
        cv2.destroyAllWindows()
        if link:
            try:
                link.write(b"STOP\n")
            finally:
                link.close()


if __name__ == "__main__":
    main()
