"""
Thumb bend calculation with live-adjustable settings.

Several candidate thumb signals ("features") are measured every frame and each
one is calibrated separately (open pose / fist pose -> 0-100 %). The final
thumb value is a weighted blend of those percentages, followed by a remap
(open point, closed point, curve). Because the blend happens in percent space,
you can change weights live without recalibrating.

Defaults reproduce the original behaviour (IP joint angle only).
No MediaPipe needed here.
"""
import copy
import json
import math

import numpy as np

# key, label, unit. Landmarks: 0 wrist, 1 CMC, 2 MCP, 3 IP, 4 tip,
# 5 index base, 9 middle base, 17 pinky base.
FEATURES = (
    ("ip", "IP joint angle", "deg"),
    ("mcp", "MCP joint angle", "deg"),
    ("cmc", "CMC joint angle", "deg"),
    ("spread", "Thumb vs index-bone angle", "deg"),
    ("tip_index", "Tip to index base", "palm"),
    ("tip_pinky", "Tip to pinky base", "palm"),
)
KEYS = [f[0] for f in FEATURES]

# Minimum open-vs-fist difference for a feature to be usable.
# deg: degrees; palm: fraction of wrist->middle-base length.
MIN_SPAN = {"deg": 12.0, "palm": 0.10}

DEFAULTS = {
    "weights": {k: (100.0 if k == "ip" else 0.0) for k in KEYS},
    "lo": 0.0,      # blended % at/below this reads 0
    "hi": 100.0,    # blended % at/above this reads 100
    "gamma": 1.0,   # <1 bends sooner in the mid range, >1 later
}

PRESETS = (
    ("ip", "IP only (original)", {"ip": 100}),
    ("ip_mcp", "IP + MCP", {"ip": 50, "mcp": 50}),
    ("joints", "All 3 joints", {"ip": 40, "mcp": 40, "cmc": 20}),
    ("opposition", "Opposition (tip to pinky)", {"tip_pinky": 100}),
    ("mix", "Joints + opposition",
     {"ip": 30, "mcp": 30, "tip_pinky": 40}),
)
PRESET_LIST = [{"id": p[0], "label": p[1]} for p in PRESETS]


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("Degenerate landmarks")
    return v / n


def _angle(u, v):
    c = float(np.clip(np.dot(_unit(u), _unit(v)), -1, 1))
    return math.degrees(math.acos(c))


def thumb_features(points):
    """Raw feature values (same order as FEATURES) from 21 world landmarks."""
    p = np.asarray(points, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError("Invalid landmarks")
    palm = np.linalg.norm(p[9] - p[0])
    if palm < 1e-8:
        raise ValueError("Degenerate landmarks")

    def bend(a, b, c):  # 0 deg = straight
        return _angle(p[b] - p[a], p[c] - p[b])

    return np.array([
        bend(2, 3, 4),
        bend(1, 2, 3),
        bend(0, 1, 2),
        _angle(p[4] - p[1], p[5] - p[0]),
        np.linalg.norm(p[4] - p[5]) / palm,
        np.linalg.norm(p[4] - p[17]) / palm,
    ])


def _num(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(hi, max(lo, v))


def sanitize(data, base=None):
    """Merge possibly-partial, untrusted settings onto base, clamped."""
    out = copy.deepcopy(base or DEFAULTS)
    if not isinstance(data, dict):
        return out
    w = data.get("weights")
    if isinstance(w, dict):
        for k in KEYS:
            if k in w:
                out["weights"][k] = _num(w[k], 0, 100, out["weights"][k])
    out["lo"] = _num(data.get("lo", out["lo"]), 0, 95, out["lo"])
    out["hi"] = _num(data.get("hi", out["hi"]), 5, 100, out["hi"])
    out["gamma"] = _num(data.get("gamma", out["gamma"]), 0.3, 3.0,
                        out["gamma"])
    if out["hi"] < out["lo"] + 1:
        out["hi"] = out["lo"] + 1
    return out


def preset_settings(preset_id, base):
    for pid, _, weights in PRESETS:
        if pid == preset_id:
            out = copy.deepcopy(base)
            out["weights"] = {k: float(weights.get(k, 0)) for k in KEYS}
            return out
    return None


def load(path):
    try:
        return sanitize(json.loads(path.read_text()))
    except Exception:
        return copy.deepcopy(DEFAULTS)


def save(path, settings):
    try:
        path.write_text(json.dumps(settings, indent=2))
    except Exception as e:
        print("Could not save thumb settings:", e)


def compute(feats, f_open, f_closed, settings):
    """Per-feature live values plus blended and final thumb percentage."""
    rows, num, den = [], 0.0, 0.0
    for i, (key, label, unit) in enumerate(FEATURES):
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
        lo, hi = settings["lo"], settings["hi"]
        t = min(1.0, max(0.0, (base - lo) / (hi - lo)))
        output = 100.0 * t ** settings["gamma"]
    return {
        "features": rows,
        "base": None if base is None else round(base, 1),
        "output": output,
    }
