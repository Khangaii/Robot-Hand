"""
Per-finger bend calculation with live-adjustable settings.

Several candidate signals ("features") are measured for every finger each
frame. Each is calibrated separately (open pose / fist pose -> 0-100 %), and
the finger's value is a weighted blend of those percentages followed by a
remap (open point, closed point, curve). Because blending happens in percent
space, weights can change live with no recalibration.

Default for every finger except the thumb = original behaviour
(PIP joint angle only). The thumb defaults to the values tuned by hand.
Shared by app.py (web UI) and camera_hand_mid50_v2.py (standalone). No
MediaPipe needed here.
"""
import copy
import json
import math

import numpy as np

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
BASE = {"Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}  # MCP landmark

# key, label, unit ("deg" angle, "palm" = fraction of wrist->middle-base)
THUMB_FEATURES = (
    ("ip", "IP joint angle", "deg"),
    ("mcp", "MCP joint angle", "deg"),
    ("cmc", "CMC joint angle", "deg"),
    ("spread", "Thumb vs index-bone angle", "deg"),
    ("tip_index", "Tip to index base", "palm"),
    ("tip_pinky", "Tip to pinky base", "palm"),
)
FINGER_FEATURES = (
    ("pip", "PIP joint angle", "deg"),
    ("dip", "DIP joint angle", "deg"),
    ("mcp", "MCP joint angle", "deg"),
    ("tip_wrist", "Tip to wrist", "palm"),
    ("tip_base", "Tip to own base", "palm"),
)
MIN_SPAN = {"deg": 12.0, "palm": 0.10}


def features_for(name):
    return THUMB_FEATURES if name == "Thumb" else FINGER_FEATURES


def _original(name):
    keys = [f[0] for f in features_for(name)]
    return {"weights": {k: (100.0 if i == 0 else 0.0)
                        for i, k in enumerate(keys)},
            "lo": 0.0, "hi": 100.0, "gamma": 1.0}


# Thumb values tuned by hand in the live UI.
TUNED_THUMB = {
    "weights": {"ip": 50.0, "mcp": 100.0, "cmc": 0.0, "spread": 0.0,
                "tip_index": 20.0, "tip_pinky": 75.0},
    "lo": 10.0, "hi": 80.0, "gamma": 1.0,
}

# id, label, weights, optional extra settings
THUMB_PRESETS = (
    ("tuned", "My tuned values", TUNED_THUMB["weights"],
     {"lo": 10.0, "hi": 80.0, "gamma": 1.0}),
    ("ip", "IP only (original)", {"ip": 100}, {}),
    ("ip_mcp", "IP + MCP", {"ip": 50, "mcp": 50}, {}),
    ("joints", "All 3 joints", {"ip": 40, "mcp": 40, "cmc": 20}, {}),
    ("opposition", "Opposition (tip to pinky)", {"tip_pinky": 100}, {}),
    ("mix", "Joints + opposition", {"ip": 30, "mcp": 30, "tip_pinky": 40}, {}),
)
FINGER_PRESETS = (
    ("pip", "PIP only (original)", {"pip": 100}, {}),
    ("pip_dip", "PIP + DIP", {"pip": 50, "dip": 50}, {}),
    ("joints", "All 3 joints", {"pip": 40, "dip": 30, "mcp": 30}, {}),
    ("curl", "Fingertip curl (tip to wrist)", {"tip_wrist": 100}, {}),
    ("mix", "Joints + tip curl",
     {"pip": 30, "dip": 20, "mcp": 20, "tip_wrist": 30}, {}),
)


def presets_for(name):
    return THUMB_PRESETS if name == "Thumb" else FINGER_PRESETS


def preset_list(name):
    return [{"id": p[0], "label": p[1]} for p in presets_for(name)]


def default_settings():
    s = {n: _original(n) for n in FINGER_NAMES}
    s["Thumb"] = copy.deepcopy(TUNED_THUMB)
    return s


def original_finger(name):
    return _original(name)


# ------------------------------------------------------------- measuring
def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("Degenerate landmarks")
    return v / n


def _angle(u, v):
    c = float(np.clip(np.dot(_unit(u), _unit(v)), -1, 1))
    return math.degrees(math.acos(c))


def hand_features(points):
    """{finger: np.array of raw feature values} from 21 world landmarks."""
    p = np.asarray(points, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError("Invalid landmarks")
    palm = np.linalg.norm(p[9] - p[0])
    if palm < 1e-8:
        raise ValueError("Degenerate landmarks")

    def bend(a, b, c):  # 0 deg = straight
        return _angle(p[b] - p[a], p[c] - p[b])

    def dist(a, b):
        return np.linalg.norm(p[a] - p[b]) / palm

    out = {"Thumb": np.array([
        bend(2, 3, 4), bend(1, 2, 3), bend(0, 1, 2),
        _angle(p[4] - p[1], p[5] - p[0]),
        dist(4, 5), dist(4, 17),
    ])}
    for name, b in BASE.items():
        out[name] = np.array([
            bend(b, b + 1, b + 2),
            bend(b + 1, b + 2, b + 3),
            bend(0, b, b + 1),
            dist(b + 3, 0),
            dist(b + 3, b),
        ])
    return out


def hand_vector(feats):
    return np.concatenate([feats[n] for n in FINGER_NAMES])


def split_vector(vec):
    out, i = {}, 0
    for n in FINGER_NAMES:
        k = len(features_for(n))
        out[n] = np.array(vec[i:i + k], dtype=float)
        i += k
    return out


def compute(name, feats, f_open, f_closed, settings):
    """Live per-feature values plus blended and final % for one finger."""
    rows, num, den = [], 0.0, 0.0
    for i, (key, label, unit) in enumerate(features_for(name)):
        raw = None if feats is None else float(feats[i])
        pct = None
        if raw is not None and f_open is not None and f_closed is not None:
            span = float(f_closed[i] - f_open[i])
            if abs(span) >= MIN_SPAN[unit]:
                pct = float(np.clip((raw - f_open[i]) / span * 100, 0, 100))
        w = settings["weights"][key]
        if pct is not None and w > 0:
            num += w * pct
            den += w
        rows.append({
            "key": key, "label": label, "unit": unit,
            "raw": None if raw is None else round(raw, 3),
            "pct": None if pct is None else round(pct, 1),
        })
    base = output = None
    if den > 0:
        base = num / den
        t = min(1.0, max(0.0, (base - settings["lo"])
                         / (settings["hi"] - settings["lo"])))
        output = 100.0 * t ** settings["gamma"]
    return {"features": rows,
            "base": None if base is None else round(base, 1),
            "output": output}


# -------------------------------------------------------------- settings
def _num(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(v) else min(hi, max(lo, v))


def sanitize_finger(name, data, base=None):
    """Merge possibly-partial, untrusted settings onto base, clamped."""
    out = copy.deepcopy(base or _original(name))
    if not isinstance(data, dict):
        return out
    w = data.get("weights")
    if isinstance(w, dict):
        for k in out["weights"]:
            if k in w:
                out["weights"][k] = _num(w[k], 0, 100, out["weights"][k])
    out["lo"] = _num(data.get("lo", out["lo"]), 0, 95, out["lo"])
    out["hi"] = _num(data.get("hi", out["hi"]), 5, 100, out["hi"])
    out["gamma"] = _num(data.get("gamma", out["gamma"]), 0.3, 3.0,
                        out["gamma"])
    if out["hi"] < out["lo"] + 1:
        out["hi"] = out["lo"] + 1
    return out


def apply_preset(name, preset_id, base):
    for pid, _, weights, extra in presets_for(name):
        if pid == preset_id:
            out = copy.deepcopy(base)
            out["weights"] = {f[0]: float(weights.get(f[0], 0))
                              for f in features_for(name)}
            out.update(extra)
            return sanitize_finger(name, out, out)
    return None


def load_settings(path):
    base = default_settings()
    try:
        data = json.loads(path.read_text())
        return {n: sanitize_finger(n, data.get(n), base[n])
                for n in FINGER_NAMES}
    except Exception:
        return base


def save_settings(path, settings):
    try:
        path.write_text(json.dumps(settings, indent=2))
    except Exception as e:
        print("Could not save tuning settings:", e)


# ----------------------------------------------------------- calibration
def load_calibration(path):
    """(open, closed) as {finger: array}, or (None, None) if missing/old."""
    try:
        d = json.loads(path.read_text())
        o, c = {}, {}
        for n in FINGER_NAMES:
            k = len(features_for(n))
            o[n] = np.array(d["open"][n], dtype=float)
            c[n] = np.array(d["closed"][n], dtype=float)
            if len(o[n]) != k or len(c[n]) != k:
                raise ValueError("shape")
        return o, c
    except Exception:
        return None, None


def save_calibration(path, cal_open, cal_closed):
    try:
        path.write_text(json.dumps({
            "open": {n: cal_open[n].tolist() for n in FINGER_NAMES},
            "closed": {n: cal_closed[n].tolist() for n in FINGER_NAMES},
        }))
    except Exception as e:
        print("Could not save calibration:", e)
