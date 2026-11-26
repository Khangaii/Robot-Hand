"""Local game logic; in board mode, host it on Windows over SSH."""
import argparse
import json
import random
import socket
import subprocess
import threading
import time
import re
import serial
from game_link import PORT, encode
from huang_game import Round, POSES, BEATS, classify, verdict


class ThrowRound(Round):
    """Commit the player's pose, robot motion and verdict together."""
    counters = {loser: winner for winner, loser in BEATS.items()}
    pair_window = 0.25

    def __init__(self, counter=False):
        super().__init__()
        self.counter = counter

    def start(self, now):
        super().start(now)
        self.throw_at = self.deadline
        self.previous = None
        self.previous_time = None
        self.player = None

    def tick(self, now, acquired, fingers):
        if self.state == "countdown" and now >= self.throw_at:
            self.state = "show"

        if self.state not in ("countdown", "show", "capture"):
            return
        if (self.player is None and self.throw_at - self.pair_window <= acquired <= now
                and (self.state != "capture" or acquired <= self.deadline)
                and (self.previous_time is None or acquired > self.previous_time)):
            label = classify(fingers)
            if (label and acquired >= self.throw_at and label == self.previous
                    and acquired - self.previous_time <= self.pair_window):
                self.player = label
                self.pose = self.counters[label] if self.counter else random.choice(tuple(POSES))
            self.previous = label
            self.previous_time = acquired

        if self.state == "capture" and now >= self.deadline:
            self.message = "Try again"
            if self.player:
                self.message = " / ".join((self.player, self.pose, verdict(self.player, self.pose)))
            self.state = "result"

    def lines(self, now):
        lines = super().lines(now)
        if self.state == "result" and self.player:
            lines[-1] = "Space: next round"
        return lines


class Engine:
    def __init__(self):
        self.game = Round()

    def close(self):
        self.game = Round()

    def frame(self, event):
        now, acquired, fingers, action, shown = event
        g = self.game
        if action == "reset":
            self.close()
            g = self.game
        if shown is not None:
            g.displayed(shown)
        if action in ("start", "start2") and g.state in ("ready", "result") and fingers is not None:
            self.game = g = ThrowRound(counter=action == "start2")
            g.start(now)
        g.tick(now, acquired, fingers)
        return dict(state=g.state, lines=g.lines(now), output=POSES.get(g.pose))


def serve(server, tunnel, stop):
    server.settimeout(1)
    while tunnel.poll() is None and not stop.is_set():
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        engine = Engine()
        with conn, conn.makefile("rb") as stream:
            try:
                conn.settimeout(15)
                while line := stream.readline(4096):
                    conn.sendall(encode(engine.frame(json.loads(line))))
            except (OSError, ValueError, TypeError):
                pass


def forward_motors(uno, stop):
    status = None
    while not stop.is_set():
        try:
            with socket.create_connection(("127.0.0.1", PORT + 1), timeout=2) as conn:
                conn.settimeout(2)
                with conn.makefile("rb") as stream:
                    while not stop.is_set():
                        packet = stream.readline(64)
                        if not packet:
                            break
                        if not re.fullmatch(rb"(?:HOME|STOP|(?:100|[1-9]?[0-9])(?:,(?:100|[1-9]?[0-9])){4})\n", packet):
                            raise ValueError("Invalid motor packet")
                        uno.write(packet)
                        if status != "connected":
                            print("Receiving D3-G commands ->", uno.port, flush=True)
                            status = "connected"
        except serial.SerialException as exc:
            print("Arduino serial error:", exc, flush=True)
            stop.set()
            return
        except (OSError, ValueError):
            pass
        if status == "connected":
            try:
                uno.write(b"HOME\n")
            except serial.SerialException as exc:
                print("Arduino serial error:", exc, flush=True)
                stop.set()
                return
        if status != "waiting":
            print("Waiting for D3-G camera stream...", flush=True)
            status = "waiting"
        stop.wait(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("board_ip")
    parser.add_argument("--port", required=True, help="Arduino COM port on Windows")
    args = parser.parse_args()
    with serial.Serial(args.port, 115200, write_timeout=0.5) as uno, socket.socket() as server:
        time.sleep(2)  # Opening the COM port may reset the UNO.
        print("Arduino open:", uno.port, flush=True)
        server.bind(("127.0.0.1", PORT))
        server.listen(1)
        command = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=10",
                   "-L", f"127.0.0.1:{PORT+1}:127.0.0.1:{PORT+1}",
                   "-R", f"127.0.0.1:{PORT}:127.0.0.1:{PORT}", "topst@" + args.board_ip]
        tunnel = subprocess.Popen(command)
        stop = threading.Event()
        worker = threading.Thread(target=forward_motors, args=(uno, stop))
        worker.start()
        try:
            print("PC game host: keep this window open; enter SSH password if requested.", flush=True)
            serve(server, tunnel, stop)
        finally:
            stop.set()
            if tunnel.poll() is None:
                tunnel.terminate()
            tunnel.wait()
            worker.join()
            if uno.is_open:
                try:
                    uno.write(b"HOME\n")
                except serial.SerialException as exc:
                    print("Could not send HOME:", exc, flush=True)
