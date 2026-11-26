from collections import Counter
# Thumb, index, middle, ring, pinky. Percentages use the UNO's calibrated travel.
POSES = {
    "ROCK": (100, 100, 100, 100, 100),
    "PAPER": (0, 0, 0, 0, 0),
    "SCISSORS": (100, 0, 0, 100, 100),
}
BEATS = {"ROCK": "SCISSORS", "PAPER": "ROCK", "SCISSORS": "PAPER"}
PATTERNS = {tuple(v[1:]): k for k, v in POSES.items()}


def classify(values):
    """values: bend % for thumb, index, middle, ring, pinky.
    (Index..pinky alone, 4 values, still works but cannot use the thumb rule.)"""
    if values is None:
        return None
    thumb = values[0] if len(values) == 5 else None
    # Index..pinky decide rock / paper / classic scissors; the thumb is ignored.
    pattern = tuple(0 if v <= 30 else 100 if v >= 65 else None for v in values[-4:])
    label = PATTERNS.get(pattern)
    if label:
        return label
    # Extended thumb + index with the other three curled also counts as scissors.
    if thumb is not None and thumb <= 30 and pattern == (0, 100, 100, 100):
        return "SCISSORS"
    return None


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



import math
import random


class Round:
    def __init__(self):
        self.state = "ready"
        self.pose = None
        self.votes = []
        self.message = ""

    def start(self, now):
        self.state = "countdown"
        self.pose = None
        self.deadline = now + 3
        self.votes.clear()
        self.message = ""

    def tick(self, now, acquired, fingers):
        if self.state == "countdown" and now >= self.deadline:
            self.state = "show"
            self.pose = random.choice(tuple(POSES))
        elif self.state == "capture":
            if self.cue <= acquired <= self.deadline:
                self.votes.append(classify(fingers))
            if now >= self.deadline:
                player = consensus(self.votes)
                self.message = (player + " / " + self.pose + " / " + verdict(player, self.pose)
                                if player else "Try again")
                self.state = "result"

    def displayed(self, now):
        if self.state == "show":
            self.state = "capture"
            self.cue = now
            self.deadline = now + 0.5

    def lines(self, now):
        if self.state == "countdown":
            return [str(max(1, math.ceil(self.deadline - now)))]
        if self.state in ("show", "capture"):
            return ["SHOW!"]
        if self.state == "result":
            parts = self.message.split(" / ")
            if len(parts) == 3:
                return ["You: " + parts[0], "HUANG: " + parts[1], parts[2], "Space: play again"]
            return ["Try again", "Space: play again"]
        return ["READY", "Space: start"]
