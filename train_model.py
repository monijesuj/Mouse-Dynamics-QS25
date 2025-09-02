import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

BUNDLE = "artifacts.pkl"
RESAMPLE_N = 32

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

def _flatten_xy(row, x_cols, y_cols):
    xr = row[x_cols].to_numpy(float)
    yr = row[y_cols].to_numpy(float)
    return np.concatenate([xr, yr])

def _engineer_extra_features(X_df, x_cols, y_cols, proto=None, class_order=None):
    extra = []
    dist_cols = []
    for idx, row in X_df.iterrows():
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

        if proto is not None and class_order is not None:
            v = np.concatenate([xr, yr])
            for uid in class_order:
                d = np.linalg.norm(v - proto[uid])
                feats[f"dist_to_user{uid}"] = float(d)
        extra.append(feats)

    X_extra = pd.DataFrame(extra, index=X_df.index)
    if proto is not None and class_order is not None:
        dist_cols = [f"dist_to_user{uid}" for uid in class_order]
    return pd.concat([X_df, X_extra], axis=1), list(X_extra.columns), dist_cols

def _compute_prototypes(X_train_df, y_train, x_cols, y_cols):
    df = X_train_df.copy()
    df["_y"] = y_train
    protos = {}
    for uid, grp in df.groupby("_y"):
        vecs = np.stack([_flatten_xy(r, x_cols, y_cols) for _, r in grp.iterrows()], axis=0)
        protos[int(uid)] = vecs.mean(axis=0)
    df.drop(columns=["_y"], inplace=True)
    return protos

def _choose_cv(y):
    vc = pd.Series(y).value_counts()
    m = int(vc.min())
    if m >= 3:
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    elif m == 2:
        return StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    else:
        return StratifiedShuffleSplit(n_splits=8, test_size=0.3, random_state=42)

def _prepare(features_csv="features.csv"):
    df = pd.read_csv(features_csv).fillna(0.0)
    train = df[df["split"] == "train"].copy()
    test  = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("No train or test data in features.csv")

    X_train = train.drop(columns=["user", "session", "split"])
    y_train = train["user"].astype(int).values
    X_test  = test.drop(columns=["user", "session", "split"])
    y_test  = test["user"].astype(int).values

    cols = list(X_train.columns)
    x_cols = _sorted_cols(cols, "x_res_")
    y_cols = _sorted_cols(cols, "y_res_")
    assert len(x_cols) == RESAMPLE_N and len(y_cols) == RESAMPLE_N, \
        f"Expected {RESAMPLE_N} x_res_ and {RESAMPLE_N} y_res_ columns."

    return X_train, y_train, X_test, y_test, cols, x_cols, y_cols

def train_and_evaluate():
    X_train, y_train, X_test, y_test, base_cols, x_cols, y_cols = _prepare()

    class_order = sorted(np.unique(y_train).tolist())
    prototypes = _compute_prototypes(X_train, y_train, x_cols, y_cols)

    X_train_aug, extra_cols, dist_cols = _engineer_extra_features(X_train.copy(), x_cols, y_cols,
                                                                  proto=prototypes, class_order=class_order)
    X_test_aug,  _,         _         = _engineer_extra_features(X_test.copy(),  x_cols, y_cols,
                                                                  proto=prototypes, class_order=class_order)

    all_cols = list(X_train_aug.columns)

    cv = _choose_cv(y_train)

    candidates = {
        "rf": RandomForestClassifier(
            n_estimators=900, max_features="sqrt", class_weight="balanced_subsample",
            random_state=42
        ),
        "et": ExtraTreesClassifier(
            n_estimators=1200, max_features="sqrt", class_weight="balanced",
            random_state=42
        ),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=False))
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", multi_class="multinomial"))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3, weights="distance", p=2))
        ]),
    }

    grids = {
        "rf": {"max_depth": [None, 16, 24], "min_samples_leaf": [1, 3, 5]},
        "et": {"max_depth": [None, 16, 24], "min_samples_leaf": [1, 3, 5]},
        "svm_rbf": {"clf__C": [0.5, 2.0, 8.0], "clf__gamma": ["scale", 0.2, 0.05]},
        "logreg": {"clf__C": [0.2, 1.0, 5.0]},
        "knn": {"clf__n_neighbors": [1, 3, 5]},
    }

    best_name, best_model, best_cv = None, None, -1.0
    for name, est in candidates.items():
        gs = GridSearchCV(est, grids[name], cv=cv, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train_aug, y_train)
        print(f"[{name}] best CV acc={gs.best_score_:.3f} params={gs.best_params_}")
        if gs.best_score_ > best_cv:
            best_cv, best_name, best_model = gs.best_score_, name, gs.best_estimator_

    best_model.fit(X_train_aug, y_train)

    y_pred = best_model.predict(X_test_aug)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nSelected model: {best_name}  |  Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump({
        "model": best_model,
        "feature_columns": all_cols,
        "base_columns": base_cols,
        "x_cols": _sorted_cols(all_cols, "x_res_"),
        "y_cols": _sorted_cols(all_cols, "y_res_"),
        "extra_cols": extra_cols,
        "dist_cols": dist_cols,
        "class_order": class_order,
        "prototypes": prototypes,
        "resample_n": RESAMPLE_N,
        "champion": best_name
    }, BUNDLE)
    print(f"✔ Saved best model ({best_name}) + prototypes to {BUNDLE}")


if __name__ == "__main__":
    train_and_evaluate()