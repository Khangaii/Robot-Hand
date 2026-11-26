from pathlib import Path

import numpy as np


class FollowScreen:
    def __init__(self):
        with np.load(Path(__file__).with_name("huang_art.npz")) as art:
            self.art = dict(art)
        self.source_size = None

    def render(self, frame, values, status, count, states, mode):
        screen = self.art[mode].copy()
        h, w = frame.shape[:2]
        if (h, w) != self.source_size:
            scale = min(573 / w, 317 / h)
            self.width = max(1, round(w * scale))
            self.height = max(1, round(h * scale))
            self.rows = np.arange(self.height) * h // self.height
            self.cols = np.arange(self.width) * w // self.width
            self.source_size = (h, w)
        x = 34 + (573 - self.width) // 2
        y = 129 + (317 - self.height) // 2
        screen[y:y+self.height, x:x+self.width] = frame[self.rows[:, None], self.cols]
        screen[477:519, 32:507] = self.art[status]
        if status.startswith("hold_"):
            screen[479:517, 326:401] = self.art["counts"][count]
        if mode == "game":
            return screen
        for i in range(5):
            y = 158 + i * 61
            value = 101 if values is None else values[i]
            screen[y-16:y+16, 844:918] = self.art["numbers"][value]
            if values is not None and states[i] in ("OPEN", "MIDDLE", "CLOSED"):
                screen[y-15:y+15, 744:842] = self.art[states[i]]
            if value < 101:
                screen[y+18:y+26, 654:654+round(263*value/100)] = (100,117,8)
        return screen


def instruction(raw, opened, closed, capture, measured):
    if raw is None:
        return "hand"
    if capture:
        return "hold_" + capture
    if opened is None:
        return "open"
    if closed is None:
        return "closed"
    return "ready" if measured is not None else "redo"
