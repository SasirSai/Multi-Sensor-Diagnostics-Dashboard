import os
import json
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
from nptdms import TdmsFile
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis, skew
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from collections import defaultdict
import warnings

# Suppress annoying scipy kurtosis warnings on flat signals
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation')


def extract_features(signal):
    """Extract advanced statistical, harmonic, and spectral features from a 1D time-series signal."""
    if len(signal) == 0:
        return [0.0] * 20
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return [0.0] * 20
        
    mean_val = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    kurt = float(kurtosis(signal))
    skw = float(skew(signal))
    ptp = np.ptp(signal)
    
    # Advanced Diagnostics Features (Time-Domain)
    peak = np.max(np.abs(signal))
    mean_abs = np.mean(np.abs(signal))
    sqr_mean_root = np.mean(np.sqrt(np.abs(signal)))**2
    
    crest_factor = peak / rms if rms > 0 else 0.0
    shape_factor = rms / mean_abs if mean_abs > 0 else 0.0
    impulse_factor = peak / mean_abs if mean_abs > 0 else 0.0
    clearance_factor = peak / sqr_mean_root if sqr_mean_root > 0 else 0.0
    
    # Frequency domain features (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    spectral_energy = np.sum(fft_vals**2) / len(fft_vals) if len(fft_vals) > 0 else 0.0
    
    # Spectral Entropy, Centroid, and Spread
    if len(fft_vals) > 0:
        psd = fft_vals**2 / (np.sum(fft_vals**2) + 1e-12)
        spectral_entropy = -np.sum(psd * np.log2(psd + 1e-12))
        freqs = np.arange(len(fft_vals))
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-12)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_vals) / (np.sum(fft_vals) + 1e-12))
    else:
        spectral_entropy = 0.0
        spectral_centroid = 0.0
        spectral_spread = 0.0
    
    # Extract Top 3 distinct harmonic peaks (using peak separation to avoid adjacent bin leakage)
    top3_freqs = [0.0, 0.0, 0.0]
    top3_amps = [0.0, 0.0, 0.0]
    
    if len(fft_vals) > 10:
        fft_vals_copy = fft_vals.copy()
        fft_vals_copy[0] = 0.0  # Skip DC
        
        for k in range(3):
            idx = np.argmax(fft_vals_copy)
            amp = fft_vals_copy[idx]
            if amp == 0.0:
                break
            top3_freqs[k] = float(idx)
            top3_amps[k] = float(amp)
            
            # Zero out neighborhood around the peak to ensure next peak is distinct
            start_idx = max(0, idx - 10)
            end_idx = min(len(fft_vals_copy), idx + 11)
            fft_vals_copy[start_idx:end_idx] = 0.0
    
    return [float(mean_val), float(std), float(rms), kurt, skw, float(ptp), float(spectral_energy),
            float(crest_factor), float(shape_factor), float(impulse_factor), float(clearance_factor),
            float(spectral_entropy), float(spectral_centroid), float(spectral_spread),
            top3_freqs[0], top3_amps[0], top3_freqs[1], top3_amps[1], top3_freqs[2], top3_amps[2]]


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, "Dataset")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(BASE_DIR), "data")

    vibration_dir = os.path.join(dataset_dir, "vibration")
    acoustic_dir = os.path.join(dataset_dir, "acoustic")
    current_temp_dir = os.path.join(dataset_dir, "current,temp")

    classes = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']

    base_feats = [
        'Mean', 'StdDev', 'RMS', 'Kurtosis', 'Skewness', 'P2P', 'SpectralEnergy', 
        'CrestFactor', 'ShapeFactor', 'ImpulseFactor', 'ClearanceFactor',
        'SpectralEntropy', 'SpectralCentroid', 'SpectralSpread',
        'Top1_Freq', 'Top1_Amp', 'Top2_Freq', 'Top2_Amp', 'Top3_Freq', 'Top3_Amp'
    ]
    columns = (
        [f"Vib_{f}" for f in base_feats]
        + [f"Acoustic_{f}" for f in base_feats]
        + ['Acoustic_Missing']
    )

    # ------------------------------------------------------------------ #
    # 1.  STRATIFIED FILE-LEVEL SPLIT (by Class & Torque)                #
    # ------------------------------------------------------------------ #
    print("Performing Stratified File-Level Split (by Class & Torque)...")
    files_by_group = defaultdict(list)

    if not os.path.exists(vibration_dir):
        print(f"Error: Dataset not found at {vibration_dir}")
        return

    for filename in os.listdir(vibration_dir):
        if not filename.endswith('.mat'):
            continue
        label = 'Unknown'
        for c in classes:
            if c in filename:
                label = c
                break
        if 'Unbalalnce' in filename:
            label = 'Unbalance'

        torque = filename.split('_')[0]  # '0Nm', '2Nm', '4Nm'
        group_key = f"{label}_{torque}"
        files_by_group[group_key].append(filename)

    train_files = set()
    test_files = set()
    random.seed(42)

    for group_key, files in files_by_group.items():
        files.sort()
        random.shuffle(files)
        n_test = max(1, int(len(files) * 0.2)) if len(files) > 1 else 0
        for f in files[:n_test]:
            test_files.add(f)
        for f in files[n_test:]:
            train_files.add(f)

    print(f"File Split Complete: {len(train_files)} Train files, {len(test_files)} Test files")

    # ------------------------------------------------------------------ #
    # 2.  FEATURE EXTRACTION                                              #
    # ------------------------------------------------------------------ #
    X_train_list, y_train_list, groups_train_list = [], [], []
    X_test_list, y_test_list = [], []

    print("Parsing Multi-Sensor Data and Extracting 20-Feature Vectors...")

    columns_initialized = False

    for filename in os.listdir(vibration_dir):
        if not filename.endswith('.mat'):
            continue

        label = 'Unknown'
        for c in classes:
            if c in filename:
                label = c
                break
        if 'Unbalalnce' in filename:
            label = 'Unbalance'

        vib_path = os.path.join(vibration_dir, filename)
        acous_path = os.path.join(acoustic_dir, filename)
        tdms_filename = filename.replace('.mat', '.tdms')

        if not os.path.exists(os.path.join(current_temp_dir, tdms_filename)):
            if 'Unbalalnce' in tdms_filename:
                tdms_filename = tdms_filename.replace('Unbalalnce', 'Unbalance')

        tdms_path = os.path.join(current_temp_dir, tdms_filename)

        try:
            vib_mat = loadmat(vib_path)
            vib_signal = (
                vib_mat['Signal']['y_values'][0, 0]['values'][0, 0].flatten()
                if 'Signal' in vib_mat else np.array([])
            )

            if os.path.exists(acous_path):
                acous_mat = loadmat(acous_path)
                acous_signal = (
                    acous_mat['Signal']['y_values'][0, 0]['values'][0, 0].flatten()
                    if 'Signal' in acous_mat else np.array([])
                )
                acoustic_missing = 0.0
            else:
                acous_signal = np.array([])
                acoustic_missing = 1.0

            tdms_file = TdmsFile.read(tdms_path)
            tdms_group = tdms_file.groups()[1]
            tdms_channels = tdms_group.channels()

            vib_chunk_size = 10000
            num_windows = len(vib_signal) // vib_chunk_size
            if num_windows == 0:
                continue

            tdms_chunk_sizes = [len(ch.data) // num_windows for ch in tdms_channels]

            max_chunks_per_file = 1000
            actual_windows = min(num_windows, max_chunks_per_file)

            for i in range(actual_windows):
                row_features = []

                v_idx = i * vib_chunk_size
                row_features.extend(
                    extract_features(vib_signal[v_idx:v_idx + vib_chunk_size])
                )
                row_features.extend(
                    extract_features(acous_signal[v_idx:v_idx + vib_chunk_size])
                )
                row_features.append(acoustic_missing)

                for ch_idx, ch in enumerate(tdms_channels):
                    t_chunk = tdms_chunk_sizes[ch_idx]
                    t_idx = i * t_chunk
                    ch_signal = ch.data[t_idx:t_idx + t_chunk]
                    row_features.extend(extract_features(ch_signal))

                    if not columns_initialized and i == 0:
                        columns.extend(
                            [f"TDMS_Ch{ch_idx}_{f}" for f in base_feats]
                        )

                if not columns_initialized:
                    columns_initialized = True

                if len(row_features) == len(columns):
                    if filename in test_files:
                        X_test_list.append(row_features)
                        y_test_list.append(label)
                    else:
                        X_train_list.append(row_features)
                        y_train_list.append(label)
                        groups_train_list.append(filename)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(
        f"Extraction Complete.  Train samples: {len(X_train_list)}, "
        f"Test samples: {len(X_test_list)}"
    )

    if len(X_train_list) == 0 or len(X_test_list) == 0:
        print("Error: Could not extract enough data for training or testing.")
        return

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    print(f"Feature dimension per sample: {X_train.shape[1]}")

    # ------------------------------------------------------------------ #
    # 3.  HYPERPARAMETER TUNING  (baseline grid, 3-fold GroupKFold)       #
    # ------------------------------------------------------------------ #
    print("\nPerforming Hyperparameter Search "
          "(RandomizedSearchCV, 3-fold GroupKFold, 12 iterations)...")

    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [20, 25, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    groups_train = np.array(groups_train_list)
    gkf = GroupKFold(n_splits=3)

    clf = RandomizedSearchCV(
        rf, param_grid,
        n_iter=12,
        cv=gkf,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train, y_train, groups=groups_train)

    print(f"\nBest Parameters Found: {clf.best_params_}")
    print(f"Best CV F1-macro: {clf.best_score_ * 100:.2f}%")
    best_clf = clf.best_estimator_

    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    report_dct = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nFinal Genuine Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report (on Unseen Hold-out Files across all Torques):")
    print(report_str)

    # ------------------------------------------------------------------ #
    # 4.  SAVE METADATA AND MODEL WEIGHTS                                #
    # ------------------------------------------------------------------ #
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [columns[i].replace("_", " ") for i in indices if i < len(columns)]

    # Convert numpy types in best_params_ for JSON serialisation
    best_params_clean = {}
    for k, v in clf.best_params_.items():
        if isinstance(v, (np.integer,)):
            best_params_clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            best_params_clean[k] = float(v)
        else:
            best_params_clean[k] = v

    analytics = {
        "accuracy": round(acc * 100, 2),
        "total_samples": len(X_train) + len(X_test),
        "top_features": top_features,
        "classes": list(best_clf.classes_)
    }

    # Persist weights
    joblib.dump(best_clf, os.path.join(model_dir, "rf_model.joblib"))
    joblib.dump(columns[:X_train.shape[1]], os.path.join(model_dir, "feature_names.joblib"))
    joblib.dump(best_clf.classes_, os.path.join(model_dir, "classes.joblib"))

    with open(os.path.join(model_dir, "analytics.json"), "w") as f:
        json.dump(analytics, f)

    # Also persist test-file list alongside
    with open(os.path.join(model_dir, "test_set_files.json"), "w") as f:
        json.dump(sorted(list(test_files)), f, indent=4)

    print(f"\nAll models and analytics saved to: {model_dir}")


if __name__ == "__main__":
    main()
