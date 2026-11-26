"""One request/reply per game frame over the loopback SSH tunnel."""
import json
import socket
import time

PORT = 29371


def encode(value):
    return json.dumps(value, allow_nan=False).encode() + b"\n"


class Client:
    def __init__(self):
        self.sock = self.stream = None
        self.retry = 0

    def close(self):
        if self.stream:
            self.stream.close()
        if self.sock:
            self.sock.close()
        self.sock = self.stream = None

    def frame(self, event):
        now = time.monotonic()
        if not self.sock and now < self.retry:
            return None
        try:
            if not self.sock:
                self.sock = socket.create_connection(("127.0.0.1", PORT), 0.5)
                self.stream = self.sock.makefile("rb")
            self.sock.sendall(encode(event))
            line = self.stream.readline(4096)
            if not line:
                raise ConnectionError("Game host disconnected")
            return json.loads(line)
        except (OSError, ValueError):
            self.close()
            self.retry = now + 1
            return None
