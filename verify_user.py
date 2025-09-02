import sys
import joblib
import numpy as np
import pandas as pd
from extract_features import extract_features_from_session

BUNDLE = "artifacts.pkl"

def _sorted_cols(cols, prefix):
    pairs = []
    for c in cols:
        if c.startswith(prefix):
            try:
                idx = int(c.split("_")[-1])
            except Exception:
                continue
            pairs.append((idx, c))
    return [c for _, c in sorted(pairs)]

def _angle_bins(dx, dy, n_bins=8):
    ang = np.arctan2(dy, dx)
    b = np.floor((ang + np.pi) / (2 * np.pi) * n_bins).astype(int)
    b = np.clip(b, 0, n_bins - 1)
    return b, ang

def _hist_norm(ids, n):
    h = np.bincount(ids, minlength=n).astype(float)
    s = h.sum()
    return h / s if s > 0 else h

def _bigram_hist(ids, n):
    if len(ids) < 2:
        return np.zeros(n * n, dtype=float)
    idx = ids[:-1] * n + ids[1:]
    h = np.bincount(idx, minlength=n * n).astype(float)
    s = h.sum()
    return h / s if s > 0 else h

def _turn_inflection_rate(ang):
    if len(ang) < 3:
        return 0.0
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    s = np.sign(d)
    changes = np.sum(s[:-1] * s[1:] < 0)
    return changes / (len(d) - 1)

def _quadrant_props(xr, yr):
    if len(xr) == 0:
        return np.zeros(4, dtype=float)
    q0 = np.mean((xr >= 0) & (yr >= 0))
    q1 = np.mean((xr <  0) & (yr >= 0))
    q2 = np.mean((xr <  0) & (yr <  0))
    q3 = np.mean((xr >= 0) & (yr <  0))
    return np.array([q0, q1, q2, q3], dtype=float)

def _engineer_from_bundle(x_df, bundle):
    cols = bundle["feature_columns"]
    x_cols = _sorted_cols(cols, "x_res_")
    y_cols = _sorted_cols(cols, "y_res_")
    class_order = bundle["class_order"]
    prototypes = bundle["prototypes"]

    extras = []
    for _, row in x_df.iterrows():
        xr = row[x_cols].to_numpy(float)
        yr = row[y_cols].to_numpy(float)
        dx = np.diff(xr)
        dy = np.diff(yr)
        bins, ang = _angle_bins(dx, dy, n_bins=8)
        dir_h = _hist_norm(bins, 8)
        bi_h = _bigram_hist(bins, 8)
        infl = _turn_inflection_rate(ang)
        quads = _quadrant_props(xr, yr)
        feats = {
            **{f"dir8_{i}": dir_h[i] for i in range(8)},
            **{f"dir8x8_{i}": bi_h[i] for i in range(64)},
            "turn_inflect_rate": float(infl),
            "quad_q0": float(quads[0]),
            "quad_q1": float(quads[1]),
            "quad_q2": float(quads[2]),
            "quad_q3": float(quads[3]),
        }
        v = np.concatenate([xr, yr])
        for uid in class_order:
            d = np.linalg.norm(v - prototypes[uid])
            feats[f"dist_to_user{uid}"] = float(d)
        extras.append(feats)

    extra_df = pd.DataFrame(extras, index=x_df.index)
    return pd.concat([x_df, extra_df], axis=1)

def verify_session(session_path: str, bundle_path: str = BUNDLE) -> int:
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    train_cols = bundle["feature_columns"]

    s = extract_features_from_session(session_path)
    if s is None:
        raise ValueError("Session too short or contains no usable motion rows.")
    x = pd.DataFrame([s]).fillna(0.0)

    for c in train_cols:
        if c not in x.columns and (c.startswith("x_res_") or c.startswith("y_res_")):
            x[c] = 0.0

    x_aug = _engineer_from_bundle(x, bundle)

    for c in train_cols:
        if c not in x_aug.columns:
            x_aug[c] = 0.0
    x_aug = x_aug[train_cols]

    pred = int(model.predict(x_aug.values)[0])
    print(f"Predicted User: {pred}")
    return pred

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_user.py <path/to/session>")
        sys.exit(1)
    verify_session(sys.argv[1])