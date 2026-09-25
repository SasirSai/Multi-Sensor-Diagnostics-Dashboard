import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.io import loadmat
from nptdms import TdmsFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from export_model import extract_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "raw")
VIBRATION_DIR = os.path.join(DATASET_DIR, "vibration")
ACOUSTIC_DIR = os.path.join(DATASET_DIR, "acoustic")
CURRENT_TEMP_DIR = os.path.join(DATASET_DIR, "current_temp")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEST_SET_PATH = os.path.join(MODEL_DIR, "test_set_files.json")
RESULTS_JSON = os.path.join(MODEL_DIR, "sensor_ablation_results.json")
RESULTS_CSV = os.path.join(MODEL_DIR, "sensor_ablation_results.csv")

CLASSES = ["Normal", "BPFI", "BPFO", "Misalign", "Unbalance"]
BASE_FEATURE_COUNT = 20
VIBRATION_CHUNK_SIZE = 10000
MAX_CHUNKS_PER_FILE = 1000

PARAM_GRID = {
    "n_estimators": [200, 300, 400],
    "max_depth": [20, 25, 30, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"],
}

CONFIGURATIONS = [
    ("Vibration", True, False, False, False),
    ("Vibration + Acoustic", True, True, False, False),
    ("Vibration + Motor Current", True, False, True, False),
    ("Vibration + Temperature", True, False, False, True),
    ("All Sensors", True, True, True, True),
]


def label_for_file(filename):
    if "Unbalalnce" in filename:
        return "Unbalance"
    for label in CLASSES:
        if label in filename:
            return label
    raise ValueError(f"Could not identify class label from filename: {filename}")


def load_signal(mat_path):
    mat = loadmat(mat_path)
    if "Signal" not in mat:
        return np.array([])
    return mat["Signal"]["y_values"][0, 0]["values"][0, 0].flatten()


def load_split():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as file:
        test_files = set(json.load(file))

    all_files = {name for name in os.listdir(VIBRATION_DIR) if name.endswith(".mat")}
    if len(all_files) != 45 or len(test_files) != 12 or len(all_files - test_files) != 33:
        raise RuntimeError(
            f"Expected the established 33/12 split, found {len(all_files - test_files)}/{len(test_files)}."
        )
    if not test_files.issubset(all_files):
        raise RuntimeError("Persisted test set contains files missing from the vibration dataset.")

    return all_files - test_files, test_files


def extract_file_rows(filename, include_acoustic, include_current, include_temperature):
    vib_signal = load_signal(os.path.join(VIBRATION_DIR, filename))
    acoustic_path = os.path.join(ACOUSTIC_DIR, filename)
    acoustic_signal = load_signal(acoustic_path) if os.path.exists(acoustic_path) else np.array([])

    tdms_filename = filename.replace(".mat", ".tdms")
    if not os.path.exists(os.path.join(CURRENT_TEMP_DIR, tdms_filename)):
        tdms_filename = tdms_filename.replace("Unbalalnce", "Unbalance")
    tdms_path = os.path.join(CURRENT_TEMP_DIR, tdms_filename)

    with TdmsFile.read(tdms_path) as tdms_file:
        log_group = tdms_file["Log"]
        channels = log_group.channels()
        channel_types = [channel.properties.get("DAC~Channel~Type") for channel in channels]
        selected_indices = []
        if include_temperature:
            selected_indices.extend(
                index for index, channel_type in enumerate(channel_types) if channel_type == "Temperature"
            )
        if include_current:
            selected_indices.extend(
                index for index, channel_type in enumerate(channel_types) if channel_type == "Current"
            )

        num_windows = len(vib_signal) // VIBRATION_CHUNK_SIZE
        if num_windows == 0:
            return [], []

        tdms_chunk_sizes = [len(channel.data) // num_windows for channel in channels]
        actual_windows = min(num_windows, MAX_CHUNKS_PER_FILE)
        rows = []
        feature_names = [f"Vib_{index}" for index in range(BASE_FEATURE_COUNT)]
        if include_acoustic:
            feature_names.extend(f"Acoustic_{index}" for index in range(BASE_FEATURE_COUNT))
            feature_names.append("Acoustic_Missing")
        for index in selected_indices:
            feature_names.extend(f"TDMS_Ch{index}_{feature}" for feature in range(BASE_FEATURE_COUNT))

        for window_index in range(actual_windows):
            row = []
            vib_start = window_index * VIBRATION_CHUNK_SIZE
            row.extend(extract_features(vib_signal[vib_start:vib_start + VIBRATION_CHUNK_SIZE]))

            if include_acoustic:
                row.extend(extract_features(acoustic_signal[vib_start:vib_start + VIBRATION_CHUNK_SIZE]))
                row.append(0.0 if len(acoustic_signal) > 0 else 1.0)

            for channel_index in selected_indices:
                channel = channels[channel_index]
                chunk_size = tdms_chunk_sizes[channel_index]
                start = window_index * chunk_size
                row.extend(extract_features(channel.data[start:start + chunk_size]))

            if len(row) != len(feature_names):
                raise RuntimeError(
                    f"Feature dimension mismatch for {filename}: {len(row)} values, expected {len(feature_names)}."
                )
            rows.append(row)

    return rows, feature_names


def build_dataset(train_files, test_files, include_acoustic, include_current, include_temperature):
    x_train, y_train, groups_train = [], [], []
    x_test, y_test, groups_test = [], [], []
    feature_names = None

    for filename in sorted(train_files | test_files):
        rows, current_feature_names = extract_file_rows(
            filename, include_acoustic, include_current, include_temperature
        )
        if feature_names is None:
            feature_names = current_feature_names
        elif feature_names != current_feature_names:
            raise RuntimeError(f"Feature schema changed while processing {filename}.")

        label = label_for_file(filename)
        if filename in test_files:
            x_test.extend(rows)
            y_test.extend([label] * len(rows))
            groups_test.extend([filename] * len(rows))
        else:
            x_train.extend(rows)
            y_train.extend([label] * len(rows))
            groups_train.extend([filename] * len(rows))

    if not x_train or not x_test:
        raise RuntimeError("The fixed split produced an empty train or test matrix.")

    return (
        np.asarray(x_train), np.asarray(y_train), np.asarray(groups_train),
        np.asarray(x_test), np.asarray(y_test), np.asarray(groups_test), feature_names,
    )


def run_configuration(name, include_acoustic, include_current, include_temperature, train_files, test_files):
    x_train, y_train, groups_train, x_test, y_test, groups_test, feature_names = build_dataset(
        train_files, test_files, include_acoustic, include_current, include_temperature
    )

    previous_results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as file:
            previous_results = {
                item["configuration"]: item
                for item in json.load(file).get("results", [])
            }

    previous = previous_results.get(name)
    if previous and "best_parameters" in previous:
        # Reuse the exact parameters selected by the completed main run.
        saved_parameters = {
            key: value for key, value in previous["best_parameters"].items()
            if key in PARAM_GRID
        }
        best_model = RandomForestClassifier(
            random_state=42, n_jobs=-1, **saved_parameters
        )
        best_model.fit(x_train, y_train)
        best_cv_macro_f1 = previous.get("best_cv_macro_f1")
    else:
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            classifier,
            PARAM_GRID,
            n_iter=12,
            cv=GroupKFold(n_splits=3),
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(x_train, y_train, groups=groups_train)
        best_model = search.best_estimator_
        best_cv_macro_f1 = float(search.best_score_)

    predictions = best_model.predict(x_test)
    file_predictions = []
    file_labels = []
    for filename in sorted(test_files):
        mask = groups_test == filename
        labels, counts = np.unique(predictions[mask], return_counts=True)
        file_predictions.append(labels[np.argmax(counts)])
        file_labels.append(y_test[mask][0])

    # Grouped out-of-fold predictions are less optimistic than scoring
    # thousands of correlated windows from the same held-out runs.
    cv_predictions = np.empty_like(y_train)
    for train_indices, validation_indices in GroupKFold(n_splits=3).split(
        x_train, y_train, groups_train
    ):
        fold_model = clone(best_model)
        fold_model.fit(x_train[train_indices], y_train[train_indices])
        cv_predictions[validation_indices] = fold_model.predict(x_train[validation_indices])

    report = classification_report(
        y_test, predictions, labels=CLASSES, output_dict=True, zero_division=0
    )
    file_report = classification_report(
        file_labels, file_predictions, labels=CLASSES, output_dict=True, zero_division=0
    )
    cv_report = classification_report(
        y_train, cv_predictions, labels=CLASSES, output_dict=True, zero_division=0
    )

    result = {
        "configuration": name,
        "feature_count": int(x_train.shape[1]),
        "train_files": len(train_files),
        "test_files": len(test_files),
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, labels=CLASSES, average="macro", zero_division=0)), 6),
        "precision_macro": round(float(report["macro avg"]["precision"]), 6),
        "recall_macro": round(float(report["macro avg"]["recall"]), 6),
        "file_accuracy": round(float(accuracy_score(file_labels, file_predictions)), 6),
        "file_macro_f1": round(float(f1_score(file_labels, file_predictions, labels=CLASSES, average="macro", zero_division=0)), 6),
        "file_precision_macro": round(float(file_report["macro avg"]["precision"]), 6),
        "file_recall_macro": round(float(file_report["macro avg"]["recall"]), 6),
        "grouped_cv_accuracy": round(float(accuracy_score(y_train, cv_predictions)), 6),
        "grouped_cv_macro_f1": round(float(f1_score(y_train, cv_predictions, labels=CLASSES, average="macro", zero_division=0)), 6),
        "grouped_cv_precision_macro": round(float(cv_report["macro avg"]["precision"]), 6),
        "grouped_cv_recall_macro": round(float(cv_report["macro avg"]["recall"]), 6),
        "best_cv_macro_f1": round(float(best_cv_macro_f1), 6),
        "feature_names": feature_names,
        "best_parameters": best_model.get_params(deep=False),
    }
    print(
        f"{name}: dimensions={result['feature_count']}, "
        f"window_accuracy={result['accuracy']:.4f}, file_accuracy={result['file_accuracy']:.4f}, "
        f"grouped_cv_f1={result['grouped_cv_macro_f1']:.4f}, "
        f"precision={result['precision_macro']:.4f}, recall={result['recall_macro']:.4f}"
    )
    return result


def main():
    train_files, test_files = load_split()
    print(f"Using fixed persisted split: {len(train_files)} train files, {len(test_files)} test files")
    print("Test files:", ", ".join(sorted(test_files)))

    results = []
    for name, _, include_acoustic, include_current, include_temperature in CONFIGURATIONS:
        results.append(
            run_configuration(
                name, include_acoustic, include_current, include_temperature, train_files, test_files
            )
        )

    metadata = {
        "split": {
            "train_file_count": len(train_files),
            "test_file_count": len(test_files),
            "train_files": sorted(train_files),
            "test_files": sorted(test_files),
            "grouping_variable": "source filename",
            "test_set_source": "models/test_set_files.json",
        },
        "windowing": {
            "vibration_chunk_size": VIBRATION_CHUNK_SIZE,
            "max_chunks_per_file": MAX_CHUNKS_PER_FILE,
        },
        "tdms_channel_mapping": {
            "temperature": "TDMS channels identified by DAC~Channel~Type=Temperature",
            "motor_current": "TDMS channels identified by DAC~Channel~Type=Current",
        },
        "results": results,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "Sensor Configuration", "Feature Dimensions", "Window Accuracy", "Window Macro-F1",
            "File Accuracy", "File Macro-F1", "Grouped CV Accuracy", "Grouped CV Macro-F1",
            "Precision", "Recall",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "Sensor Configuration": result["configuration"],
                "Feature Dimensions": result["feature_count"],
                "Window Accuracy": result["accuracy"],
                "Window Macro-F1": result["macro_f1"],
                "File Accuracy": result["file_accuracy"],
                "File Macro-F1": result["file_macro_f1"],
                "Grouped CV Accuracy": result["grouped_cv_accuracy"],
                "Grouped CV Macro-F1": result["grouped_cv_macro_f1"],
                "Precision": result["precision_macro"],
                "Recall": result["recall_macro"],
            })

    print(f"Saved JSON results to {RESULTS_JSON}")
    print(f"Saved table to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
