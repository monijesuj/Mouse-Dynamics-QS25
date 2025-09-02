from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

EPS = 1e-8
RESAMPLE_N = 32


def _read_session(path: Path) -> pd.DataFrame:
    """Read one session; auto-detect delimiter; normalize columns; keep needed cols."""
    df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]

    need = {"button", "state", "x", "y"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    for c in ("x", "y"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    st = df["state"].astype(str).str.lower()
    bt = df["button"].astype(str).str.lower()
    motion = df[(bt == "nobutton") & (st.isin(["move", "drag", "mousemove"]))].copy()

    return df, motion

def _resample_path(x: np.ndarray, y: np.ndarray, n: int = RESAMPLE_N) -> tuple[np.ndarray, np.ndarray] | None:
    """Resample polyline by arc length to `n` points; translation+scale+rotation normalized."""
    if len(x) < 3:
        return None
    x = x - x[0]
    y = y - y[0]

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)
    total = float(step.sum())
    if not np.isfinite(total) or total < EPS:
        return None

    s = np.concatenate([[0.0], np.cumsum(step)])
    s /= s[-1]

    grid = np.linspace(0.0, 1.0, n)
    xr = np.interp(grid, s, x)
    yr = np.interp(grid, s, y)

    ux, uy = xr[-1] - xr[0], yr[-1] - yr[0]
    ang = np.arctan2(uy, ux)
    ca, sa = np.cos(-ang), np.sin(-ang)
    xrot = xr * ca - yr * sa
    yrot = xr * sa + yr * ca

    xrot /= (total + EPS)
    yrot /= (total + EPS)
    return xrot, yrot

def _direction_and_turn(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ang = np.arctan2(dy, dx)
    d_ang = np.diff(ang)
    d_ang = (d_ang + np.pi) % (2 * np.pi) - np.pi
    return ang[1:], d_ang

def _hist(vals: np.ndarray, bins: int, range_tuple: tuple[float,float]) -> np.ndarray:
    h, _ = np.histogram(vals, bins=bins, range=range_tuple)
    h = h.astype(float)
    h /= (h.sum() + EPS)
    return h


def extract_features_from_session(path: str | Path) -> pd.Series | None:
    full, motion = _read_session(Path(path))
    if motion.empty or len(motion) < 3:
        return None

    x = motion["x"].to_numpy(np.float64)
    y = motion["y"].to_numpy(np.float64)

    res = _resample_path(x, y, RESAMPLE_N)
    if res is None:
        return None
    xr, yr = res
    flat_xy = np.concatenate([xr, yr])

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)
    path_len = float(step.sum())
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    straightness = disp / (path_len + EPS)

    bbox_w = float(x.max() - x.min())
    bbox_h = float(y.max() - y.min())

    _, d_ang = _direction_and_turn(dx, dy)
    dir_hist = _hist(np.arctan2(dy, dx), bins=12, range_tuple=(-np.pi, np.pi))
    turn_hist = _hist(d_ang, bins=12, range_tuple=(-np.pi, np.pi))

    curv = d_ang / (step[1:] + EPS) if len(step) > 1 else np.array([0.0])

    frac_drag = float((motion["state"].str.lower() == "drag").mean())

    def s(fn, arr, default=0.0):
        if arr.size == 0:
            return default
        v = fn(arr)
        return float(v) if np.isfinite(v) else default

    feats = {
        "n_points": int(len(x)),
        "total_distance": path_len,
        "displacement": disp,
        "straightness": straightness,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_aspect": bbox_w / (bbox_h + EPS),
        "bbox_area": bbox_w * bbox_h,
        "step_mean": s(np.mean, step),
        "step_std": s(np.std, step),
        "step_skew": s(skew, step),
        "step_kurt": s(kurtosis, step),
        "turn_abs_mean": s(lambda a: np.mean(np.abs(a)), d_ang),
        "turn_std": s(np.std, d_ang),
        "curv_mean": s(np.mean, curv),
        "curv_std": s(np.std, curv),
        "frac_drag": frac_drag,
    }

    for i, v in enumerate(dir_hist):
        feats[f"dir_hist_{i}"] = float(v)
    for i, v in enumerate(turn_hist):
        feats[f"turn_hist_{i}"] = float(v)
    for i, v in enumerate(xr):
        feats[f"x_res_{i}"] = float(v)
    for i, v in enumerate(yr):
        feats[f"y_res_{i}"] = float(v)

    return pd.Series(feats).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def load_all_data(base_dir: str | Path = "data") -> pd.DataFrame:
    base = Path(base_dir)
    rows = []
    for split_dir in ["training_files", "test_files"]:
        split_path = base / split_dir
        if not split_path.exists():
            print(f"Warning: {split_path} not found. Skipping.")
            continue
        for user_dir in sorted(split_path.glob("user*")):
            digits = "".join(ch for ch in user_dir.name if ch.isdigit())
            if not digits:
                continue
            user_id = int(digits)
            files = [p for p in user_dir.iterdir()
                     if p.is_file() and not p.name.startswith(".")
                     and p.suffix.lower() in {".csv", ".txt", ""}]
            if not files:
                print(f"  (no session files) {user_dir}")
                continue
            for f in files:
                try:
                    s = extract_features_from_session(f)
                    if s is None:
                        continue
                    s["user"] = user_id
                    s["session"] = f.name
                    s["split"] = "train" if split_dir == "training_files" else "test"
                    rows.append(s)
                except Exception as e:
                    print(f"Failed {f}: {e}")

    return pd.DataFrame(rows).fillna(0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="data")
    ap.add_argument("--out", default="features.csv")
    args = ap.parse_args()

    df = load_all_data(args.base_dir)
    if df.empty:
        print("No data loaded. Check your data structure.")
        return
    df.to_csv(args.out, index=False)
    n_train = int((df["split"] == "train").sum())
    n_test  = int((df["split"] == "test").sum())
    print(f"✔ Extracted {len(df)} sessions from {df['user'].nunique()} users "
          f"({n_train} train, {n_test} test). Saved to {args.out}")

if __name__ == "__main__":
    main()