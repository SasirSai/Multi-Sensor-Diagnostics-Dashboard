"""
sujith_model_corrected.py
========================
CORRECTED SUJITH MODEL: Random Forest + Isolation Forest (Strict Leak-Free Split)

This script trains Sujith's exact machine learning architecture (Random Forest + Isolation Forest)
under strict, publication-ready scientific validation:
1. Replaces chunk-level split with strict Stratified File-Level Split.
2. Removes the test-set target accuracy search loop and random noise perturbation loop.
3. Uses a standard GroupKFold cross-validation on training data to select hyperparameters.
4. Saves all model files and analytics strictly under models/sujith_corrected/.
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
from collections import defaultdict
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE EXTRACTION  (Sujith's exact 8 basic features)
# =============================================================================

def extract_features(signal):
    """Extract Sujith's 8 basic statistical features from a 1D time-series signal."""
    if len(signal) == 0:
        return [0.0]*8
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return [0.0]*8
        
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    
    if std > 0:
        diff = signal - mean
        skw = np.mean(diff**3) / (std**3)
        kurt = np.mean(diff**4) / (std**4) - 3.0
    else:
        skw = 0.0
        kurt = 0.0
        
    ptp = np.ptp(signal)
    
    # Frequency domain features (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    spectral_energy = np.sum(fft_vals**2) / len(fft_vals) if len(fft_vals) > 0 else 0.0
    peak_freq = np.argmax(fft_vals) if len(fft_vals) > 0 else 0.0
    
    return [mean, std, rms, kurt, skw, ptp, spectral_energy, float(peak_freq)]

# =============================================================================
# DATA LOADING  (Rigorous File-Level Stratification)
# =============================================================================

def load_data(base_dir):
    dataset_dir = os.path.join(base_dir, "Dataset")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(base_dir), "data")
    
    vibration_dir = os.path.join(dataset_dir, "vibration")
    acoustic_dir = os.path.join(dataset_dir, "acoustic")
    current_temp_dir = os.path.join(dataset_dir, "current,temp")
    
    classes = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
    
    base_feats = ['Mean', 'StdDev', 'RMS', 'Kurtosis', 'Skewness', 'P2P', 'SpectralEnergy', 'PeakFreq']
    columns = [f"Vib_{f}" for f in base_feats] + [f"Acoustic_{f}" for f in base_feats]
    
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
            
            # Acoustic is optional, zero-filled if missing (correlated fallback)
            if os.path.exists(acous_path):
                acous_mat = loadmat(acous_path)
                acous_signal = acous_mat['Signal']['y_values'][0,0]['values'][0,0].flatten() if 'Signal' in acous_mat else np.array([])
            else:
                acous_signal = np.array([])
            
            tdms_file = TdmsFile.read(tdms_path)
            tdms_group = tdms_file.groups()[1]
            tdms_channels = tdms_group.channels()
            
            vib_chunk_size = 10000 
            num_windows = len(vib_signal) // vib_chunk_size
            if num_windows == 0: continue
                
            tdms_chunk_sizes = [len(ch.data) // num_windows for ch in tdms_channels]
            
            # Support up to 1000 chunks per file to match the expanded training volume of other models
            max_chunks_per_file = 1000
            actual_windows = min(num_windows, max_chunks_per_file)
            
            for i in range(actual_windows):
                row_features = []
                
                v_idx = i * vib_chunk_size
                row_features.extend(extract_features(vib_signal[v_idx:v_idx+vib_chunk_size]))
                
                if len(acous_signal) > 0:
                    row_features.extend(extract_features(acous_signal[v_idx:v_idx+vib_chunk_size]))
                else:
                    row_features.extend(row_features[:len(base_feats)])
                
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
    model_dir = os.path.join(BASE_DIR, "models", "sujith_corrected")
    os.makedirs(model_dir, exist_ok=True)
    
    (X_train, y_train, groups_train,
     X_test, y_test,
     test_files, columns) = load_data(BASE_DIR)
     
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Insufficient data extracted.")
        return
        
    # STEP 1 — Isolation Forest trained ONLY on Normal training samples
    print("\n[Sujith Corrected] Training Isolation Forest on Normal training samples only...")
    normal_mask = y_train == 'Normal'
    X_normal = X_train[normal_mask]
    print(f"  Normal training samples available: {len(X_normal)}")
    
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_normal)
    
    # STEP 2 — GroupKFold Hyperparameter Optimization (Genuine Validation)
    print("\n[Sujith Corrected] Hyperparameter Tuning (RandomizedSearchCV + GroupKFold)...")
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [20, 25, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }
    
    gkf = GroupKFold(n_splits=3)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        n_iter=12,
        cv=gkf,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train, groups=groups_train)
    
    best_clf = search.best_estimator_
    print(f"  Best Parameters: {search.best_params_}")
    
    # STEP 3 — Evaluate on completely unseen hold-out files
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    report_dct = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n[Sujith Corrected] Final Genuine Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report (Unseen Hold-out Files, all Torques):")
    print(report_str)
    
    # Feature importance
    top5_idx = np.argsort(best_clf.feature_importances_)[::-1][:5]
    top5_feats = [columns[i].replace("_", " ") for i in top5_idx]
    
    # STEP 4 — Save strictly to models/sujith_corrected/ (completely safe)
    analytics = {
        "architecture": "Random Forest + Isolation Forest (Sujith Corrected) [Leak-free]",
        "accuracy": round(acc * 100, 2),
        "total_samples": len(X_train) + len(X_test),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": int(X_train.shape[1]),
        "best_rf_params": search.best_params_,
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
    
    joblib.dump(iso_forest, os.path.join(model_dir, "iso_model.joblib"))
    joblib.dump(best_clf, os.path.join(model_dir, "rf_model.joblib"))
    joblib.dump(best_clf.classes_, os.path.join(model_dir, "classes.joblib"))
    joblib.dump(columns, os.path.join(model_dir, "feature_names.joblib"))
    
    with open(os.path.join(model_dir, "analytics.json"), "w") as f:
        json.dump(analytics, f, indent=4)
        
    print(f"\nAll models and analytics saved to: {model_dir}")

if __name__ == "__main__":
    main()
