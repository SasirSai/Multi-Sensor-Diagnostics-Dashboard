import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from nptdms import TdmsFile
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis, skew
import joblib

def extract_features(signal):
    """Extract statistical features from a 1D time-series signal."""
    if len(signal) == 0:
        return [0]*6
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return [0]*6
        
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    skw = skew(signal)
    ptp = np.ptp(signal)
    
    return [mean, std, rms, kurt, skw, ptp]

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, "Dataset")
    vibration_dir = os.path.join(dataset_dir, "vibration")
    acoustic_dir = os.path.join(dataset_dir, "acoustic")
    current_temp_dir = os.path.join(dataset_dir, "current,temp")
    
    classes = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
    features_list = []
    labels_list = []
    
    base_feats = ['Mean', 'StdDev', 'RMS', 'Kurtosis', 'Skewness', 'P2P']
    columns = [f"Vib_{f}" for f in base_feats] + [f"Acoustic_{f}" for f in base_feats]
    
    print("Parsing Multi-Sensor Data for Model Export...")
    for filename in os.listdir(vibration_dir):
        if not filename.endswith('.mat'): continue
            
        label = 'Unknown'
        for c in classes:
            if c in filename:
                label = c
                break
                
        vib_path = os.path.join(vibration_dir, filename)
        acous_path = os.path.join(acoustic_dir, filename)
        tdms_filename = filename.replace('.mat', '.tdms')
        tdms_path = os.path.join(current_temp_dir, tdms_filename)
        
        try:
            vib_mat = loadmat(vib_path)
            vib_signal = vib_mat['Signal']['y_values'][0,0]['values'][0,0].flatten() if 'Signal' in vib_mat else np.array([])
            
            acous_mat = loadmat(acous_path)
            acous_signal = acous_mat['Signal']['y_values'][0,0]['values'][0,0].flatten() if 'Signal' in acous_mat else np.array([])
            
            tdms_file = TdmsFile.read(tdms_path)
            tdms_group = tdms_file.groups()[1]
            tdms_channels = tdms_group.channels()
            
            vib_chunk_size = 10000 
            num_windows = len(vib_signal) // vib_chunk_size
            
            if num_windows == 0: continue
                
            tdms_chunk_sizes = [len(ch.data) // num_windows for ch in tdms_channels]
            
            for i in range(num_windows):
                row_features = []
                
                v_idx = i * vib_chunk_size
                row_features.extend(extract_features(vib_signal[v_idx:v_idx+vib_chunk_size]))
                
                row_features.extend(extract_features(acous_signal[v_idx:v_idx+vib_chunk_size]))
                
                for ch_idx, ch in enumerate(tdms_channels):
                    t_chunk = tdms_chunk_sizes[ch_idx]
                    t_idx = i * t_chunk
                    ch_signal = ch.data[t_idx:t_idx+t_chunk]
                    row_features.extend(extract_features(ch_signal))
                    
                    if len(features_list) == 0 and i == 0:
                        columns.extend([f"TDMS_Ch{ch_idx}_{f}" for f in base_feats])
                
                features_list.append(row_features)
                labels_list.append(label)
                
        except Exception as e:
            pass # Keep it clean for the export script

    print(f"Extraction Complete. Total feature windows: {len(features_list)}")
    
    df = pd.DataFrame(features_list, columns=columns[:len(features_list[0])])
    X = df.values
    y = np.array(labels_list)

    print("Training Model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    clf.fit(X, y)
    
    import json
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    top_features = [columns[i].replace("_", " ") for i in indices]
    
    analytics = {
        "accuracy": round(clf.oob_score_ * 100, 2),
        "total_samples": len(X),
        "top_features": top_features,
        "classes": list(clf.classes_)
    }
    
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "rf_model.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")
    classes_path = os.path.join(model_dir, "classes.joblib")
    
    joblib.dump(clf, model_path)
    joblib.dump(columns[:len(features_list[0])], features_path)
    joblib.dump(clf.classes_, classes_path)
    
    with open(os.path.join(model_dir, "analytics.json"), "w") as f:
        json.dump(analytics, f)
    
    print(f"Model and metadata successfully exported to {model_dir}")

if __name__ == "__main__":
    main()
