from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import numpy as np
import pandas as pd
from scipy.io import loadmat
from nptdms import TdmsFile
from scipy.stats import kurtosis, skew
import joblib

app = FastAPI(title="Diagnostic API Dashboard")

# Configure CORS so the frontend can easily talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for the HTML dashboard to access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and metadata
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
try:
    clf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    classes = joblib.load(os.path.join(MODEL_DIR, "classes.joblib"))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model. Please run export_model.py first. Error: {e}")
    clf, feature_names, classes = None, None, None

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
    kurt = float(kurtosis(signal))
    skw = float(skew(signal))
    ptp = np.ptp(signal)
    
    return [float(mean), float(std), float(rms), kurt, skw, float(ptp)]

@app.get("/")
async def root():
    return {"message": "Diagnostics Engine API is Live! Please send POST requests to the /predict endpoint."}

@app.post("/predict")
async def predict(
    vib_file: UploadFile = File(...),
    acous_file: UploadFile = File(...),
    tdms_file: UploadFile = File(...),
    chunk_index: int = Form(0) # Which time chunk (0-indexed) to analyze
):
    if clf is None:
         raise HTTPException(status_code=500, detail="Model is not loaded. Run export_model.py")

    # Save uploaded files temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    vib_path = os.path.join(temp_dir, f"vib_{vib_file.filename}")
    acous_path = os.path.join(temp_dir, f"acous_{acous_file.filename}")
    tdms_path = os.path.join(temp_dir, f"tdms_{tdms_file.filename}")
    
    with open(vib_path, "wb") as buffer:
        shutil.copyfileobj(vib_file.file, buffer)
    with open(acous_path, "wb") as buffer:
        shutil.copyfileobj(acous_file.file, buffer)
    with open(tdms_path, "wb") as buffer:
        shutil.copyfileobj(tdms_file.file, buffer)

    try:
        # 1. Load Vibration
        vib_mat = loadmat(vib_path)
        vib_signal = vib_mat['Signal']['y_values'][0,0]['values'][0,0].flatten() if 'Signal' in vib_mat else np.array([])
        
        # 2. Load Acoustic
        acous_mat = loadmat(acous_path)
        acous_signal = acous_mat['Signal']['y_values'][0,0]['values'][0,0].flatten() if 'Signal' in acous_mat else np.array([])
        
        # 3. Load Current & Temp (.tdms)
        tdms_file_loaded = TdmsFile.read(tdms_path)
        tdms_group = tdms_file_loaded.groups()[1] # Log group
        tdms_channels = tdms_group.channels()

        vib_chunk_size = 10000 
        num_windows = len(vib_signal) // vib_chunk_size
        
        if num_windows == 0: 
            raise ValueError("Vibration signal is too short to extract one chunk.")
            
        tdms_chunk_sizes = [len(ch.data) // num_windows for ch in tdms_channels]

        # Use the requested chunk (or default to 0)
        idx = min(chunk_index, num_windows - 1)
        
        row_features = []
        
        # Extract Vib
        v_idx = idx * vib_chunk_size
        row_features.extend(extract_features(vib_signal[v_idx:v_idx+vib_chunk_size]))
        
        # Extract Acoustic
        row_features.extend(extract_features(acous_signal[v_idx:v_idx+vib_chunk_size]))
        
        # Extract TDMS Channels
        for ch_idx, ch in enumerate(tdms_channels):
            t_chunk = tdms_chunk_sizes[ch_idx]
            t_idx = idx * t_chunk
            ch_signal = ch.data[t_idx:t_idx+t_chunk]
            row_features.extend(extract_features(ch_signal))
            
        # Format for prediction
        X = np.array([row_features])
        
        # Predict
        prediction = clf.predict(X)[0]
        probabilities = clf.predict_proba(X)[0]
        
        # Clean up
        os.remove(vib_path)
        os.remove(acous_path)
        os.remove(tdms_path)
        
        # Create response payload
        response = {
            "prediction": str(prediction),
            "probabilities": {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probabilities)},
            "features": {name: round(float(val), 4) for name, val in zip(feature_names, row_features)}
        }
        
        return response

    except Exception as e:
        # Clean up on error
        if os.path.exists(vib_path): os.remove(vib_path)
        if os.path.exists(acous_path): os.remove(acous_path)
        if os.path.exists(tdms_path): os.remove(tdms_path)
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
