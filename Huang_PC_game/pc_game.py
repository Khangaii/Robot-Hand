"""Local game logic; in board mode, host it on Windows over SSH."""
import argparse
import json
import socket
import subprocess
from game_link import PORT, encode
from huang_game import Round, POSES, BEATS, classify, verdict


class CounterRound(Round):
    """Lock one observed throw, and use it for both the servo pose and score."""
    counters = {loser: winner for winner, loser in BEATS.items()}

    def start(self, now):
        super().start(now)
        self.previous = None
        self.previous_time = None
        self.player = None

    def tick(self, now, acquired, fingers):
        if self.state not in ("countdown", "capture"):
            return
        if self.state == "countdown":
            if now < self.deadline - 0.25:
                return
            if now >= self.deadline:
                self.state = "show"
        within_window = self.state != "capture" or self.cue <= acquired <= self.deadline
        if within_window and not self.player:
            label = classify(fingers)
            fresh_pair = (self.previous_time is not None
                          and 0 < acquired - self.previous_time <= 0.25)
            if (self.state != "countdown" and label and fresh_pair
                    and label == self.previous):
                self.player = label
                self.pose = self.counters[label]
            self.previous = label
            self.previous_time = acquired
        if self.state == "capture" and now >= self.deadline:
            self.message = "Try again"
            if self.player:
                self.message = " / ".join((self.player, self.pose, verdict(self.player, self.pose)))
            self.state = "result"


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
            self.game = g = CounterRound() if action == "start2" else Round()
            g.start(now)
        g.tick(now, acquired, fingers)
        return dict(state=g.state, lines=g.lines(now), output=POSES.get(g.pose))


def serve(server, tunnel):
    server.settimeout(1)
    while tunnel.poll() is None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("board_ip")
    args = parser.parse_args()
    with socket.socket() as server:
        server.bind(("127.0.0.1", PORT))
        server.listen(1)
        command = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=10",
                   "-R", f"127.0.0.1:{PORT}:127.0.0.1:{PORT}", "topst@" + args.board_ip]
        tunnel = subprocess.Popen(command)
        try:
            print("PC game host: keep this window open; enter SSH password if requested.", flush=True)
            serve(server, tunnel)
        finally:
            if tunnel.poll() is None:
                tunnel.terminate()
            tunnel.wait()
