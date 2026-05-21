import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from nptdms import TdmsFile
from scipy.stats import kurtosis, skew
import joblib

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    deps_available = True
except ImportError:
    deps_available = False

# =============================================================================
# FEATURE EXTRACTION METHODS
# =============================================================================

def extract_features_upgraded(signal):
    """Extract advanced statistical, harmonic, and spectral features (20 features)."""
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
            
            # Zero out neighborhood around the peak
            start_idx = max(0, idx - 10)
            end_idx = min(len(fft_vals_copy), idx + 11)
            fft_vals_copy[start_idx:end_idx] = 0.0
    
    return [float(mean_val), float(std), float(rms), kurt, skw, float(ptp), float(spectral_energy),
            float(crest_factor), float(shape_factor), float(impulse_factor), float(clearance_factor),
            float(spectral_entropy), float(spectral_centroid), float(spectral_spread),
            top3_freqs[0], top3_amps[0], top3_freqs[1], top3_amps[1], top3_freqs[2], top3_amps[2]]

def extract_features_sujith(signal):
    """Extract Sujith's 8 basic statistical features."""
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
# DATA EXTRACTION FOR TEST FILES
# =============================================================================

def extract_test_data(base_dir, test_files):
    """Parse raw datasets strictly for the isolated test holdout files."""
    dataset_dir = os.path.join(base_dir, "Dataset")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(base_dir), "data")
        
    vibration_dir = os.path.join(dataset_dir, "vibration")
    acoustic_dir = os.path.join(dataset_dir, "acoustic")
    current_temp_dir = os.path.join(dataset_dir, "current,temp")
    
    classes = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
    
    # Pre-allocate containers
    X_test_upgraded = []
    X_test_sujith = []
    y_test_labels = []
    
    print(f"Extracting features from {len(test_files)} isolated test files...")
    
    for filename in test_files:
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
            
            # Slice holdout set chunks
            actual_windows = min(num_windows, 1000)
            
            for i in range(actual_windows):
                # 1. UPGRADED FEATURE VECTOR (20 per sensor + missing acoustic flag)
                row_up = []
                v_idx = i * vib_chunk_size
                row_up.extend(extract_features_upgraded(vib_signal[v_idx:v_idx+vib_chunk_size]))
                row_up.extend(extract_features_upgraded(acous_signal[v_idx:v_idx+vib_chunk_size]))
                row_up.append(acoustic_missing)
                
                # 2. SUJITH FEATURE VECTOR (8 per sensor)
                row_suj = []
                row_suj.extend(extract_features_sujith(vib_signal[v_idx:v_idx+vib_chunk_size]))
                if len(acous_signal) > 0:
                    row_suj.extend(extract_features_sujith(acous_signal[v_idx:v_idx+vib_chunk_size]))
                else:
                    row_suj.extend(row_suj[:8]) # Replicate vib if missing
                
                # Process TDMS channels
                for ch_idx, ch in enumerate(tdms_channels):
                    t_chunk = tdms_chunk_sizes[ch_idx]
                    t_idx = i * t_chunk
                    ch_signal = ch.data[t_idx:t_idx+t_chunk]
                    
                    row_up.extend(extract_features_upgraded(ch_signal))
                    row_suj.extend(extract_features_sujith(ch_signal))
                    
                X_test_upgraded.append(row_up)
                X_test_sujith.append(row_suj)
                y_test_labels.append(label)
                
        except Exception as e:
            print(f"Failed to process test file {filename}: {e}")
            
    return np.array(X_test_upgraded), np.array(X_test_sujith), np.array(y_test_labels)

# =============================================================================
# CONFUSION MATRIX PLOTTING
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Draw a highly readable, annotated, journal-quality confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Calculate percentages for cell text
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100
    
    labels = []
    for i in range(len(classes)):
        row = []
        for j in range(len(classes)):
            val = cm[i, j]
            perc = cm_perc[i, j]
            row.append(f"{val}\n({perc:.1f}%)")
        labels.append(row)
        
    labels = np.array(labels)
    
    plt.figure(figsize=(8, 7), dpi=300)
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes,
                linewidths=1.0, linecolor="#CBD5E1",
                annot_kws={"size": 11, "weight": "bold", "color": "#0F172A"})
                
    plt.title(title, fontsize=13, weight="bold", pad=15, color="#0F172A")
    plt.ylabel("Actual State Class", fontsize=11, color="#334155")
    plt.xlabel("Predicted State Class", fontsize=11, color="#334155")
    plt.xticks(fontsize=10, color="#334155")
    plt.yticks(fontsize=10, color="#334155", rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf", format="pdf")
    plt.close()

# =============================================================================
# DEEP ANALYTICS REPORT GENERATOR
# =============================================================================

def calculate_advanced_metrics(y_true, y_pred, classes):
    """Compute exhaustive diagnostic statistics (Sensitivity, Specificity, FPR, FNR)."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics = {}
    
    for idx, cls in enumerate(classes):
        # TP, FP, FN, TN calculation from multi-class confusion matrix
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        # Sensitivity (Recall)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Specificity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # False Positive Rate (FPR)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # False Negative Rate (FNR)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        # F1-Score
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        
        metrics[cls] = {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Sensitivity (Recall)": round(sens * 100, 2),
            "Specificity": round(spec * 100, 2),
            "Precision": round(prec * 100, 2),
            "FPR": round(fpr * 100, 2),
            "FNR": round(fnr * 100, 2),
            "F1-Score": round(f1 * 100, 2)
        }
    return metrics

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    if not deps_available:
        print("Dependency Error: matplotlib, seaborn, and scikit-learn are required to run this script.")
        return
        
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "models")
    output_dir = os.path.join(model_dir, "visual_analytics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load holdout test files list
    test_files_path = os.path.join(model_dir, "test_set_files.json")
    if not os.path.exists(test_files_path):
        print(f"Error: test_set_files.json not found at {test_files_path}")
        return
    with open(test_files_path, "r") as f:
        test_files = json.load(f)
        
    # Extract test features once
    X_test_upgraded, X_test_sujith, y_test = extract_test_data(BASE_DIR, test_files)
    print(f"Test samples extracted: {len(y_test)}")
    
    # Paths mapping
    models_paths = {
        "Optimized Hybrid (RF + IF)": {
            "clf": os.path.join(model_dir, "hybrid_optimized", "rf_model_hybrid.joblib"),
            "iso": os.path.join(model_dir, "hybrid_optimized", "isolation_forest.joblib"),
            "features_type": "upgraded_hybrid",
            "save_name": "confusion_matrix_hybrid"
        },
        "Upgraded Pure RF": {
            "clf": os.path.join(model_dir, "rf_model.joblib"),
            "features_type": "upgraded_pure",
            "save_name": "confusion_matrix_pure_rf"
        },
        "Corrected Sujith RF + IF": {
            "clf": os.path.join(model_dir, "sujith_corrected", "rf_model.joblib"),
            "features_type": "sujith",
            "save_name": "confusion_matrix_sujith"
        },
        "Optimized GB + IF": {
            "clf": os.path.join(model_dir, "gradient_boost_optimized", "gb_model.joblib"),
            "iso": os.path.join(model_dir, "gradient_boost_optimized", "isolation_forest.joblib"),
            "features_type": "upgraded_hybrid",
            "save_name": "confusion_matrix_gb"
        }
    }
    
    classes = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
    reports = {}
    
    for name, config in models_paths.items():
        if not os.path.exists(config["clf"]):
            print(f"Skipping {name}: model weight not found at {config['clf']}.")
            continue
            
        print(f"\nEvaluating Model: {name}...")
        clf = joblib.load(config["clf"])
        
        # Build features depending on model type
        if config["features_type"] == "upgraded_hybrid":
            iso = joblib.load(config["iso"])
            anomaly_scores = iso.decision_function(X_test_upgraded).reshape(-1, 1)
            X_eval = np.hstack([X_test_upgraded, anomaly_scores])
        elif config["features_type"] == "upgraded_pure":
            X_eval = X_test_upgraded
        else: # sujith
            X_eval = X_test_sujith
            
        # Predict
        y_pred = clf.predict(X_eval)
        
        # Plot Confusion Matrix
        title_str = f"Confusion Matrix: {name}\n(Unseen Holdout Files - Strict Split)"
        save_path = os.path.join(output_dir, config["save_name"])
        plot_confusion_matrix(y_test, y_pred, classes, title_str, save_path)
        print(f"  Confusion Matrix exported to: {save_path}.png/.pdf")
        
        # Calculate Advanced Diagnostics Analytics
        reports[name] = calculate_advanced_metrics(y_test, y_pred, classes)
        
    # Write the high-impact Advanced Analytics Markdown Report
    report_path = os.path.join(output_dir, "advanced_analytics_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🔬 Advanced Diagnostics Performance Analytics Report\n\n")
        f.write("This report details the exhaustive statistical classification metrics computed across all four diagnostic models on completely unseen, leak-free holdout test motor runs.\n\n")
        
        for model_name, metrics in reports.items():
            f.write(f"## 📊 Model Engine: {model_name}\n\n")
            f.write("| Fault Category | TP | TN | FP | FN | Sensitivity (Recall) % | Specificity % | Precision % | FPR % | FNR % | F1-Score % |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for cls in classes:
                m = metrics[cls]
                f.write(f"| **{cls}** | {m['TP']} | {m['TN']} | {m['FP']} | {m['FN']} | {m['Sensitivity (Recall)']} | {m['Specificity']} | {m['Precision']} | {m['FPR']} | {m['FNR']} | {m['F1-Score']} |\n")
            f.write("\n---\n\n")
            
    print(f"\nExhaustive statistical metrics compiled inside: {report_path}")

if __name__ == "__main__":
    main()
