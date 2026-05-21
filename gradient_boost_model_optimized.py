"""
gradient_boost_model_optimized.py
================================
OPTIMIZED GRADIENT BOOSTING ARCHITECTURE: HistGradientBoosting + Isolation Forest (Feature Injection)

This model incorporates:
1. Upgraded feature vector (20 features per sensor, including Spectral Entropy, Centroid, and Spread).
2. Physically sound distinct harmonic peak extraction (peak separation of 10 bins).
3. Expanded sample budget (max 1000 chunks per file).
4. Strictly isolated model outputs to models/gradient_boost_optimized/.
"""

import os
import json
import random
import numpy as np
import warnings
from scipy.io import loadmat
from nptdms import TdmsFile
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from scipy.stats import kurtosis, skew
from collections import defaultdict
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(signal):
    """Extract advanced statistical, harmonic, and spectral features from a 1D time-series signal."""
    if len(signal) == 0:
        return [0.0]*20
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return [0.0]*20
        
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

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(base_dir):
    dataset_dir = os.path.join(base_dir, "Dataset")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(base_dir), "data")
    
    vibration_dir = os.path.join(dataset_dir, "vibration")
    acoustic_dir = os.path.join(dataset_dir, "acoustic")
    current_temp_dir = os.path.join(dataset_dir, "current,temp")
    
    classes = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
    
    base_feats = ['Mean', 'StdDev', 'RMS', 'Kurtosis', 'Skewness', 'P2P', 'SpectralEnergy', 
                  'CrestFactor', 'ShapeFactor', 'ImpulseFactor', 'ClearanceFactor',
                  'SpectralEntropy', 'SpectralCentroid', 'SpectralSpread',
                  'Top1_Freq', 'Top1_Amp', 'Top2_Freq', 'Top2_Amp', 'Top3_Freq', 'Top3_Amp']
    columns = [f"Vib_{f}" for f in base_feats] + [f"Acoustic_{f}" for f in base_feats] + ['Acoustic_Missing']
    
    # Stratified File-Level Split by Class AND Torque
    print("Performing Stratified File-Level Split (by Class & Torque)...")
    files_by_group = defaultdict(list)
    
    for filename in os.listdir(vibration_dir):
        if not filename.endswith('.mat'): continue
        label = 'Unknown'
        for c in classes:
            if c in filename:
                label = c
                break
        if 'Unbalalnce' in filename:
            label = 'Unbalance'
            
        torque = filename.split('_')[0]
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
    
    X_train, y_train, groups_train = [], [], []
    X_test, y_test = [], []
    
    print("Parsing Multi-Sensor Data and Extracting Advanced Features...")
    columns_initialized = False
    
    for filename in os.listdir(vibration_dir):
        if not filename.endswith('.mat'): continue
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
            vib_signal = vib_mat['Signal']['y_values'][0,0]['values'][0,0].flatten() if 'Signal' in vib_mat else np.array([])
            
            if os.path.exists(acous_path):
                acous_mat = loadmat(acous_path)
                acous_signal = acous_mat['Signal']['y_values'][0,0]['values'][0,0].flatten() if 'Signal' in acous_mat else np.array([])
                acoustic_missing = 0.0
            else:
                acous_signal = np.array([])
                acoustic_missing = 1.0
            
            tdms_file = TdmsFile.read(tdms_path)
            tdms_group = tdms_file.groups()[1]
            tdms_channels = tdms_group.channels()
            
            vib_chunk_size = 10000 
            num_windows = len(vib_signal) // vib_chunk_size
            if num_windows == 0: continue
                
            tdms_chunk_sizes = [len(ch.data) // num_windows for ch in tdms_channels]
            
            max_chunks_per_file = 1000
            actual_windows = min(num_windows, max_chunks_per_file)
            
            for i in range(actual_windows):
                row_features = []
                
                v_idx = i * vib_chunk_size
                row_features.extend(extract_features(vib_signal[v_idx:v_idx+vib_chunk_size]))
                row_features.extend(extract_features(acous_signal[v_idx:v_idx+vib_chunk_size]))
                row_features.append(acoustic_missing)
                
                for ch_idx, ch in enumerate(tdms_channels):
                    t_chunk = tdms_chunk_sizes[ch_idx]
                    t_idx = i * t_chunk
                    ch_signal = ch.data[t_idx:t_idx+t_chunk]
                    row_features.extend(extract_features(ch_signal))
                    
                    if not columns_initialized and i == 0:
                        columns.extend([f"TDMS_Ch{ch_idx}_{f}" for f in base_feats])
                
                if not columns_initialized:
                    columns_initialized = True
                
                if len(row_features) == len(columns):
                    if filename in test_files:
                        X_test.append(row_features)
                        y_test.append(label)
                    else:
                        X_train.append(row_features)
                        y_train.append(label)
                        groups_train.append(filename)
                
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    print(f"Extraction Complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return (np.array(X_train), np.array(y_train), np.array(groups_train),
            np.array(X_test), np.array(y_test), test_files, columns)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "models", "gradient_boost_optimized")
    os.makedirs(model_dir, exist_ok=True)
    
    (X_train, y_train, groups_train,
     X_test, y_test,
     test_files, columns) = load_data(BASE_DIR)
     
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Insufficient data extracted.")
        return
        
    # STEP 1 — Isolation Forest trained ONLY on Normal training samples
    print("\n[GB+IF] Training Isolation Forest on Normal training samples only...")
    normal_mask = y_train == 'Normal'
    X_normal = X_train[normal_mask]
    print(f"  Normal training samples available: {len(X_normal)}")
    
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_normal)
    
    # STEP 2 — Inject continuous Anomaly Score as a genuine new feature
    train_scores = iso_forest.decision_function(X_train).reshape(-1, 1)
    test_scores = iso_forest.decision_function(X_test).reshape(-1, 1)
    
    X_train_aug = np.hstack([X_train, train_scores])
    X_test_aug = np.hstack([X_test, test_scores])
    print(f"  Anomaly score injected. Total features per sample: {X_train_aug.shape[1]}")
    
    # STEP 3 — HistGradientBoostingClassifier with GroupKFold tuning
    print("\n[GB+IF] Hyperparameter Tuning (RandomizedSearchCV + GroupKFold)...")
    param_grid = {
        'max_iter': [100, 200, 300, 400],
        'max_depth': [6, 8, 10, None],
        'max_leaf_nodes': [31, 63, 127],
        'min_samples_leaf': [10, 20, 30],
        'l2_regularization': [0.0, 0.1, 0.5],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    
    base_clf = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        class_weight='balanced'
    )
    
    gkf = GroupKFold(n_splits=3)
    search = RandomizedSearchCV(
        base_clf,
        param_grid,
        n_iter=12,
        cv=gkf,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train_aug, y_train, groups=groups_train)
    
    best_clf = search.best_estimator_
    print(f"\n  Best Parameters: {search.best_params_}")
    
    # STEP 4 — Evaluate on completely unseen hold-out files
    y_pred = best_clf.predict(X_test_aug)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, zero_division=0)
    report_dct = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"\n[GB+IF] Final Genuine Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report (Unseen Hold-out Files, all Torques):")
    print(report_str)
    
    # Permutation feature importances
    all_cols = list(columns) + ['IF_AnomalyScore']
    n_perm = min(2000, len(X_test_aug))
    rng = np.random.default_rng(42)
    perm_idx = rng.choice(len(X_test_aug), size=n_perm, replace=False)
    
    print("\nCalculating permutation feature importances...")
    perm_result = permutation_importance(
        best_clf, X_test_aug[perm_idx], y_test[perm_idx],
        n_repeats=5, random_state=42, n_jobs=-1, scoring='f1_macro'
    )
    top5_idx = np.argsort(perm_result.importances_mean)[::-1][:5]
    top5_feats = [all_cols[i] if i < len(all_cols) else 'IF_AnomalyScore' for i in top5_idx]
    
    # STEP 5 — Save strictly to models/gradient_boost_optimized/ (completely safe)
    analytics = {
        "architecture": "HistGradientBoosting + Isolation Forest (Feature Injection) [Optimized]",
        "accuracy": round(acc * 100, 2),
        "total_samples": len(X_train) + len(X_test),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": int(X_train_aug.shape[1]),
        "best_gb_params": {k: str(v) for k, v in search.best_params_.items()},
        "top_5_features": top5_feats,
        "classes": list(best_clf.classes_),
        "macro_avg_f1": round(report_dct['macro avg']['f1-score'] * 100, 2),
        "per_class": {
            cls: {
                "precision": round(report_dct[cls]['precision'] * 100, 2),
                "recall": round(report_dct[cls]['recall'] * 100, 2),
                "f1": round(report_dct[cls]['f1-score'] * 100, 2)
            }
            for cls in best_clf.classes_ if cls in report_dct
        },
        "test_files": sorted(list(test_files))
    }
    
    joblib.dump(iso_forest, os.path.join(model_dir, "isolation_forest.joblib"))
    joblib.dump(best_clf, os.path.join(model_dir, "gb_model.joblib"))
    joblib.dump(best_clf.classes_, os.path.join(model_dir, "classes.joblib"))
    
    with open(os.path.join(model_dir, "gb_analytics.json"), "w") as f:
        json.dump(analytics, f, indent=4)
        
    print(f"\nAll models and analytics saved to: {model_dir}")

if __name__ == "__main__":
    main()
