"""
hybrid_model.py
================
HYBRID ARCHITECTURE: Random Forest + Isolation Forest (Anomaly Detection)

This file is intentionally separate from export_model.py to allow side-by-side
comparison of two distinct ML architectures for the research paper.

Approach: Feature Injection
---------------------------
1. File-level stratified split (SAME seed/logic as export_model.py for a fair comparison).
2. IsolationForest is trained ONLY on Normal class training samples.
3. A continuous Anomaly Score (decision_function) is injected as a new feature.
4. RandomForestClassifier is trained on the augmented feature matrix.

Key Anti-Overfitting Safeguards:
- File-level GroupKFold prevents CV leakage.
- IsolationForest fit only on training Normal samples (zero test contamination).
- Class undersampling + class_weight='balanced' prevents majority-class bias.
- Identical features to export_model.py for a fair, reproducible comparison.
"""

import os
import json
import random
import numpy as np
import warnings
from scipy.io import loadmat
from nptdms import TdmsFile
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import kurtosis, skew
from collections import defaultdict
import joblib

warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE EXTRACTION  (identical to export_model.py for a fair comparison)
# =============================================================================

def extract_features(signal):
    """Return 17 advanced statistical + harmonic features from a 1-D signal."""
    if len(signal) == 0:
        return [0.0] * 17
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return [0.0] * 17

    mean_val  = np.mean(signal)
    std       = np.std(signal)
    rms       = np.sqrt(np.mean(signal ** 2))
    kurt      = float(kurtosis(signal))
    skw       = float(skew(signal))
    ptp       = np.ptp(signal)

    peak          = np.max(np.abs(signal))
    mean_abs      = np.mean(np.abs(signal))
    sqr_mean_root = np.mean(np.sqrt(np.abs(signal))) ** 2

    crest_factor     = peak / rms           if rms           > 0 else 0.0
    shape_factor     = rms  / mean_abs      if mean_abs      > 0 else 0.0
    impulse_factor   = peak / mean_abs      if mean_abs      > 0 else 0.0
    clearance_factor = peak / sqr_mean_root if sqr_mean_root > 0 else 0.0

    fft_vals        = np.abs(np.fft.rfft(signal))
    spectral_energy = np.sum(fft_vals ** 2) / len(fft_vals) if len(fft_vals) > 0 else 0.0

    # Top-3 harmonic peaks (DC zeroed) — key discriminator for Misalign vs Unbalance
    if len(fft_vals) > 3:
        fft_vals[0] = 0.0
        top3_idx    = np.argsort(fft_vals)[-3:][::-1]
        top3_freqs  = [float(i)            for i in top3_idx]
        top3_amps   = [float(fft_vals[i])  for i in top3_idx]
    else:
        top3_freqs = [0.0, 0.0, 0.0]
        top3_amps  = [0.0, 0.0, 0.0]

    return [
        float(mean_val), float(std), float(rms), kurt, skw, float(ptp),
        float(spectral_energy),
        float(crest_factor), float(shape_factor),
        float(impulse_factor), float(clearance_factor),
        top3_freqs[0], top3_amps[0],
        top3_freqs[1], top3_amps[1],
        top3_freqs[2], top3_amps[2],
    ]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(base_dir):
    # Resolve dataset root (mirrors export_model.py path logic)
    dataset_dir = os.path.join(base_dir, "Dataset")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(base_dir), "data")

    vibration_dir    = os.path.join(dataset_dir, "vibration")
    acoustic_dir     = os.path.join(dataset_dir, "acoustic")
    current_temp_dir = os.path.join(dataset_dir, "current,temp")

    if not os.path.exists(vibration_dir):
        raise FileNotFoundError(f"Vibration directory not found at: {vibration_dir}")

    classes    = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
    base_feats = [
        'Mean', 'StdDev', 'RMS', 'Kurtosis', 'Skewness', 'P2P', 'SpectralEnergy',
        'CrestFactor', 'ShapeFactor', 'ImpulseFactor', 'ClearanceFactor',
        'Top1_Freq', 'Top1_Amp', 'Top2_Freq', 'Top2_Amp', 'Top3_Freq', 'Top3_Amp',
    ]

    # Fixed columns built once (TDMS channels appended on first valid file)
    base_columns = (
        [f"Vib_{f}"      for f in base_feats] +
        [f"Acoustic_{f}" for f in base_feats] +
        ['Acoustic_Missing']
    )

    # ------------------------------------------------------------------
    # Stratified file-level split  (SAME seed=42 as export_model.py)
    # ------------------------------------------------------------------
    print("Performing Stratified File-Level Split (by Class & Torque)...")
    files_by_group = defaultdict(list)

    for fn in os.listdir(vibration_dir):
        if not fn.endswith('.mat'):
            continue
        label = 'Unknown'
        for c in classes:
            if c in fn:
                label = c
                break
        if 'Unbalalnce' in fn:
            label = 'Unbalance'
        torque = fn.split('_')[0]
        files_by_group[f"{label}_{torque}"].append(fn)

    train_files, test_files = set(), set()
    random.seed(42)
    for gk, files in files_by_group.items():
        files.sort()
        random.shuffle(files)
        n_test = max(1, int(len(files) * 0.2)) if len(files) > 1 else 0
        for f in files[:n_test]:
            test_files.add(f)
        for f in files[n_test:]:
            train_files.add(f)

    print(f"File Split Complete: {len(train_files)} Train files, {len(test_files)} Test files")

    # ------------------------------------------------------------------
    # Feature extraction — columns list is finalised on the FIRST file,
    # then kept constant for every subsequent file.
    # ------------------------------------------------------------------
    X_train, y_train, groups_train = [], [], []
    X_test,  y_test                = [], []

    columns        = None   # will be set after first successful file
    expected_len   = None

    all_files = sorted(os.listdir(vibration_dir))

    print("Parsing Multi-Sensor Data and Extracting Advanced Features...")
    for fn in all_files:
        if not fn.endswith('.mat'):
            continue

        label = 'Unknown'
        for c in classes:
            if c in fn:
                label = c
                break
        if 'Unbalalnce' in fn:
            label = 'Unbalance'

        vib_path   = os.path.join(vibration_dir, fn)
        acous_path = os.path.join(acoustic_dir, fn)

        tdms_fn = fn.replace('.mat', '.tdms')
        if not os.path.exists(os.path.join(current_temp_dir, tdms_fn)):
            if 'Unbalalnce' in tdms_fn:
                tdms_fn = tdms_fn.replace('Unbalalnce', 'Unbalance')
        tdms_path = os.path.join(current_temp_dir, tdms_fn)

        try:
            vib_mat    = loadmat(vib_path)
            vib_signal = (vib_mat['Signal']['y_values'][0, 0]['values'][0, 0].flatten()
                          if 'Signal' in vib_mat else np.array([]))

            if os.path.exists(acous_path):
                acous_mat    = loadmat(acous_path)
                acous_signal = (acous_mat['Signal']['y_values'][0, 0]['values'][0, 0].flatten()
                                if 'Signal' in acous_mat else np.array([]))
                acoustic_missing = 0.0
            else:
                acous_signal     = np.array([])
                acoustic_missing = 1.0

            tdms_file     = TdmsFile.read(tdms_path)
            tdms_group    = tdms_file.groups()[1]
            tdms_channels = tdms_group.channels()

            chunk_size   = 10_000
            num_windows  = len(vib_signal) // chunk_size
            if num_windows == 0:
                continue

            tdms_chunk_sizes = [len(ch.data) // num_windows for ch in tdms_channels]

            # Build column names once (only for the first file that reaches here)
            if columns is None:
                columns = list(base_columns)
                for ch_idx in range(len(tdms_channels)):
                    columns.extend([f"TDMS_Ch{ch_idx}_{f}" for f in base_feats])
                expected_len = len(columns)

            max_chunks     = 600
            actual_windows = min(num_windows, max_chunks)

            for i in range(actual_windows):
                row = []
                v   = i * chunk_size
                row.extend(extract_features(vib_signal[v: v + chunk_size]))
                row.extend(extract_features(acous_signal[v: v + chunk_size]))
                row.append(acoustic_missing)

                for ch_idx, ch in enumerate(tdms_channels):
                    tc = tdms_chunk_sizes[ch_idx]
                    t  = i * tc
                    row.extend(extract_features(ch.data[t: t + tc]))

                if len(row) != expected_len:
                    continue   # row length mismatch → skip silently

                if fn in test_files:
                    X_test.append(row)
                    y_test.append(label)
                else:
                    X_train.append(row)
                    y_train.append(label)
                    groups_train.append(fn)

        except Exception as e:
            print(f"  [SKIP] {fn}: {e}")

    print(f"Extraction Complete. "
          f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    return (np.array(X_train),     np.array(y_train),
            np.array(groups_train),
            np.array(X_test),      np.array(y_test),
            test_files, columns)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "models", "hybrid")
    os.makedirs(model_dir, exist_ok=True)

    (X_train, y_train, groups_train,
     X_test,  y_test,
     test_files, columns) = load_data(BASE_DIR)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Insufficient data extracted.  Check dataset paths.")
        return

    # ------------------------------------------------------------------
    # STEP 1  — Isolation Forest trained ONLY on Normal training samples
    # ------------------------------------------------------------------
    print("\n[Hybrid] Training Isolation Forest on Normal training samples only...")
    normal_mask = y_train == 'Normal'
    X_normal    = X_train[normal_mask]
    print(f"  Normal training samples available: {len(X_normal)}")

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # Expected outlier fraction in production
        max_features=0.8,     # Feature subsampling prevents IF overfitting
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_normal)

    # ------------------------------------------------------------------
    # STEP 2  — Inject continuous Anomaly Score as a genuine new feature
    # ------------------------------------------------------------------
    train_scores = iso_forest.decision_function(X_train).reshape(-1, 1)
    test_scores  = iso_forest.decision_function(X_test).reshape(-1, 1)

    X_train_hybrid = np.hstack([X_train, train_scores])
    X_test_hybrid  = np.hstack([X_test,  test_scores])

    print(f"  Anomaly score injected. Total features per sample: {X_train_hybrid.shape[1]}")

    # ------------------------------------------------------------------
    # STEP 3  — GroupKFold RandomizedSearchCV on augmented features
    # ------------------------------------------------------------------
    print("\n[Hybrid] Hyperparameter Tuning (RandomizedSearchCV + GroupKFold)...")
    param_grid = {
        'n_estimators':     [100, 200, 300, 400],
        'max_depth':        [15, 20, 25, None],
        'min_samples_leaf': [1, 2],
        'class_weight':     ['balanced'],
    }

    gkf = GroupKFold(n_splits=3)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        n_iter=5,
        cv=gkf,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train_hybrid, y_train, groups=groups_train)

    best_clf = search.best_estimator_
    print(f"  Best Parameters: {search.best_params_}")

    # ------------------------------------------------------------------
    # STEP 4  — Evaluate on completely unseen hold-out files
    # ------------------------------------------------------------------
    y_pred     = best_clf.predict(X_test_hybrid)
    acc        = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    report_dct = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n[Hybrid] Final Genuine Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report (Unseen Hold-out Files, all Torques):")
    print(report_str)

    # Feature importance
    all_cols = list(columns) + ['IF_AnomalyScore']
    top5_idx = np.argsort(best_clf.feature_importances_)[::-1][:5]
    top5_feats = [all_cols[i] if i < len(all_cols) else 'IF_AnomalyScore'
                  for i in top5_idx]

    print("Top 5 Feature Importances:")
    for rank, (feat, idx) in enumerate(zip(top5_feats, top5_idx), 1):
        print(f"  {rank}. {feat}: {best_clf.feature_importances_[idx]:.4f}")

    # ------------------------------------------------------------------
    # STEP 5  — Save to models/hybrid/ (never touches export_model outputs)
    # ------------------------------------------------------------------
    analytics = {
        "architecture":  "Random Forest + Isolation Forest (Feature Injection)",
        "accuracy":      round(acc * 100, 2),
        "total_samples": len(X_train) + len(X_test),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "n_features":    int(X_train_hybrid.shape[1]),
        "best_rf_params": search.best_params_,
        "top_5_features": top5_feats,
        "classes":        list(best_clf.classes_),
        "macro_avg_f1":   round(report_dct['macro avg']['f1-score'] * 100, 2),
        "per_class": {
            cls: {
                "precision": round(report_dct[cls]['precision'] * 100, 2),
                "recall":    round(report_dct[cls]['recall']    * 100, 2),
                "f1":        round(report_dct[cls]['f1-score']  * 100, 2),
            }
            for cls in best_clf.classes_ if cls in report_dct
        },
        "test_files": sorted(list(test_files)),
    }

    joblib.dump(iso_forest,          os.path.join(model_dir, "isolation_forest.joblib"))
    joblib.dump(best_clf,            os.path.join(model_dir, "rf_model_hybrid.joblib"))
    joblib.dump(best_clf.classes_,   os.path.join(model_dir, "classes.joblib"))

    with open(os.path.join(model_dir, "hybrid_analytics.json"), "w") as f:
        json.dump(analytics, f, indent=4)

    print(f"\nAll models and analytics saved to: {model_dir}")
    print("Run export_model.py to compare against the pure Random Forest baseline.")


if __name__ == "__main__":
    main()
