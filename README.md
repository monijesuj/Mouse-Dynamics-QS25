## Start

0. create env

python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scipy scikit-learn joblib

1. (optional, recommended) prune test set using public_labels.csv

python3 filter_tests.py --apply

2. extract features (timestamps ignored)

python3 extract_features.py # → features.csv

3. train & evaluate (cross-validated model selection; then single pass on test set)

python3 train_model.py # → artifacts.pkl

4. predict a single session

python3 verify_user.py data/test_files/user7/session_0041905

5. verification thresholds from TRAIN only (no reliance on test labels)

python3 calibrate_thresholds.py --far 0.05 # → thresholds.json

6. batch scoring (identification / verification)

python3 score_sessions.py --input data/test_files --topk 5 --out scores_ident.csv
python3 score_sessions.py --input data/test_files/user21 --claim 21 --thresholds thresholds.json --out verify_u21.csv

## Layout

data/
training*files/
user7/ session*_ (txt/csv/or no extension)
user9/ ...
...
test*files/
user7/ session*_
...
public_labels.csv
extract_features.py
filter_tests.py
train_model.py
verify_user.py

## How the pipeline works

1. Feature extraction (extract_features.py)

For each session:
• Filter to motion only: button == NoButton and state ∈ {Move, Drag, MouseMove}.
• Resample & normalize the trajectory to a fixed-length polyline (default 32 points):
• translate to origin,
• rotate so net displacement points to +x,
• scale by total path length.
This yields x_res_0..31, y_res_0..31 (shape-only, time-independent).
• Compute compact shape descriptors:
• path stats: total distance, displacement, straightness, bounding box, curvature-like measures,
• 8-bin direction histogram and 8×8 bigram histogram (stroke turns),
• turn inflection rate, quadrant occupancy,
• (at train time only) prototype distances: Euclidean distance to each user’s mean resampled path.

The script writes one row per session to features.csv with split = 'train' or split = 'test' based on folder.

2. Model training (train_model.py)
   • Load features.csv and split by split (never train on test).
   • Compute class prototypes on TRAIN only; append engineered features + prototype distances to both train/test (no label leakage—distances use only train prototypes).
   • Model selection with adaptive cross-validation on train:
   • RandomForest, ExtraTrees, SVM (RBF), LogisticRegression, kNN.
   • Class balancing where applicable; small-data-friendly CV.
   • Fit the best model on all TRAIN features; evaluate once on TEST (diagnostic).
   • Save to artifacts.pkl: model, feature column order, prototypes, class order, etc.

3. Single-session prediction (verify_user.py)
   • Extract features for one file, rebuild engineered features using saved prototypes, align columns, and call model.predict(...).

4. Test-set cleaning (filter_tests.py)
   • Keep only sessions listed in public_labels.csv with is_illegal == 0 and remove the rest from data/test_files/\*\*.
   • Run with --apply to actually delete; without it, it’s a dry-run.

5. Verification mode (when test labels are noisy)
   • calibrate_thresholds.py builds per-user thresholds from TRAIN only targeting a chosen FAR (false-accept rate).
   • score_sessions.py can:
   • produce Top-k identification for any folder,
   • or perform verification given a claimed user id and thresholds (ACCEPT/REJECT per session).

## Results

    •	Training set size: ~65 sessions total (≈6–7 per user).
    •	Cleaned test set (legal only via public_labels.csv): 411 sessions.
    •	Selected model: Random Forest (balanced; tuned depth/leaf; ~900–1200 trees depending on grid).
    •	Test accuracy (Top-1 identification on cleaned set): 54.74% (0.5474).
    •	Per-user recalls (examples):
    •	user 7: 0.97 | user 9: 1.00 | user 21: 1.00
    •	user 12: 0.48 | user 15: 0.42 | user 16: 0.00 (hard case)
    •	Cross-validated train accuracy (label-faithful): best fold ≈ 0.43 (kNN, from CV printout).
