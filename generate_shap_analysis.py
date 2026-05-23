import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False

from generate_confusion_matrices import extract_test_data

def main():
    if not shap_available:
        print("Error: The 'shap' library is not installed. Run 'pip install shap' first.")
        return

    print("Initializing SHAP Explainable AI (XAI) Analysis...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "models")
    output_dir = os.path.join(model_dir, "visual_analytics")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Proposed Pure RF model and features
    clf_path = os.path.join(model_dir, "rf_model.joblib")
    feats_path = os.path.join(model_dir, "feature_names.joblib")
    test_files_path = os.path.join(model_dir, "test_set_files.json")

    if not (os.path.exists(clf_path) and os.path.exists(feats_path) and os.path.exists(test_files_path)):
        print("Error: Proposed Pure RF model assets not found. Run training/evaluation scripts first.")
        return

    print("Loading Proposed Pure RF model weights...")
    clf = joblib.load(clf_path)
    feature_names = list(joblib.load(feats_path))
    
    with open(test_files_path, "r") as f:
        test_files = json.load(f)

    # 2. Extract test data (using our leak-free extraction logic)
    print("Extracting test data features...")
    X_test_upgraded, _, _, y_test = extract_test_data(BASE_DIR, test_files)
    print(f"Total test instances extracted: {X_test_upgraded.shape[0]}")

    # 3. Create a representative stratified subset for fast & reliable SHAP computation
    # (SHAP on large random forests with 9684 samples and 141 features is computationally intensive)
    print("Selecting representative stratified subset of 200 samples (40 per class)...")
    np.random.seed(42)
    classes = list(clf.classes_)
    sample_indices = []
    
    for cls in classes:
        cls_indices = np.where(y_test == cls)[0]
        if len(cls_indices) > 0:
            sampled = np.random.choice(cls_indices, min(40, len(cls_indices)), replace=False)
            sample_indices.extend(sampled)
            
    X_sample = X_test_upgraded[sample_indices]
    y_sample = y_test[sample_indices]
    
    # Verify predictions match actuals on sample
    y_sample_pred = clf.predict(X_sample)
    sample_acc = accuracy_score(y_sample, y_sample_pred)
    print(f"Stratified sample accuracy: {sample_acc * 100:.2f}%")

    # Clean up feature names to look clean in academic plots
    cleaned_feature_names = [f.replace("_", " ").replace("Vib", "Vibration") for f in feature_names]

    # 4. Fit SHAP TreeExplainer
    print("Fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    
    print("Computing SHAP values (this may take up to a minute)...")
    shap_values = explainer.shap_values(X_sample)
    
    # Check shape of shap_values and convert to list of arrays if 3D numpy array
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        n_classes = shap_values.shape[2]
        shap_values_list = [shap_values[:, :, i] for i in range(n_classes)]
    else:
        shap_values_list = shap_values
    
    # 5. Plot 1: Stacked SHAP Summary Bar Chart (Global Feature Importance across all classes)
    print("Generating Figure 4: Stacked SHAP Summary Bar Chart...")
    plt.figure(figsize=(10, 7), dpi=300)
    shap.summary_plot(shap_values_list, X_sample, feature_names=cleaned_feature_names, 
                      class_names=classes, max_display=15, show=False)
    
    plt.title("Proposed Pure RF: Stacked Multiclass SHAP Feature Influence Analysis", 
              fontsize=13, weight="bold", pad=20, color="#0F172A")
    plt.xlabel("Mean Absolute SHAP Value (Impact on Model Outputs)", fontsize=11, color="#334155")
    plt.tight_layout()
    
    bar_save_path = os.path.join(output_dir, "shap_summary_bar")
    plt.savefig(bar_save_path + ".png", dpi=300)
    plt.savefig(bar_save_path + ".pdf", format="pdf")
    plt.close()
    print(f"  Saved Stacked Bar Chart to: {bar_save_path}.png/.pdf")

    # 6. Plot beeswarm for specific fault classes to show feature value directions
    # We focus on the structural defects (Unbalance, Misalignment) and bearing defect BPFI
    focus_classes = ["Unbalance", "Misalign", "BPFI"]
    
    for cls in focus_classes:
        if cls in classes:
            class_idx = classes.index(cls)
            print(f"Generating Beeswarm Plot for class: {cls}...")
            
            plt.figure(figsize=(11, 7), dpi=300)
            
            # For tree SHAP values of single class:
            shap.summary_plot(shap_values_list[class_idx], X_sample, feature_names=cleaned_feature_names,
                              max_display=12, plot_type=None, show=False)
            
            plt.title(f"SHAP Beeswarm Analysis: Diagnostic Dynamics for '{cls}' Class", 
                      fontsize=13, weight="bold", pad=20, color="#0F172A")
            plt.xlabel("SHAP Value (Impact on Prediction Probability)", fontsize=11, color="#334155")
            plt.tight_layout()
            
            beeswarm_save_path = os.path.join(output_dir, f"shap_beeswarm_{cls.lower()}")
            plt.savefig(beeswarm_save_path + ".png", dpi=300)
            plt.savefig(beeswarm_save_path + ".pdf", format="pdf")
            plt.close()
            print(f"  Saved Beeswarm Plot to: {beeswarm_save_path}.png/.pdf")

    print("\nSHAP Explainability Graphics successfully exported to:")
    print(f"  - {os.path.join(output_dir, 'shap_summary_bar.png/.pdf')}")
    for cls in focus_classes:
        print(f"  - {os.path.join(output_dir, f'shap_beeswarm_{cls.lower()}.png/.pdf')}")

if __name__ == "__main__":
    main()
