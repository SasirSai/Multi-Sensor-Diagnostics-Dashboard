# 📊 Technical Specification: Flowcharts of Architecture, Pipeline, and Decision Logic

## 1. Visual Specification
*   **Filenames**:
    *   System Architecture Flowchart: [flowchart_system_architecture.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/flowchart_system_architecture.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/flowchart_system_architecture.pdf)
    *   Diagnostic Pipeline Flowchart: [flowchart_diagnostic_pipeline.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/flowchart_diagnostic_pipeline.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/flowchart_diagnostic_pipeline.pdf)
    *   Decision Control Flowchart: [flowchart_decision_logic.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/flowchart_decision_logic.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/flowchart_decision_logic.pdf)
*   **Format**: Lossless `.png` and vector `.pdf` for direct publication integration.
*   **Aesthetics**: Beautifully aligned flowcharts with color-coded nodes (yellow for decisions, green for normal operations, orange for warnings, red for emergency shutdowns) matching the paper flow templates.

---

## 🏗️ 2. Flowchart 1: System Hardware Architecture
This diagram outlines the physical deployment stack of the multi-sensor diagnostic framework from edge sensing to mechanical safety shutoff:

1.  **Sensors (Vib/Acous/Elec)**: Accelerometers, microphones, and current transformer/thermocouple sensors capture the multi-modal physical states of the motor.
2.  **DAQ / ESP32 Gateway**: High-speed analog-to-digital converter (ADC) interfaces (such as ESP32 microcontroller boards) sample the sensor signals.
3.  **Edge CPU (Raspberry Pi/Jetson)**: The gateway streams the synchronized digital signals to an edge computer node (e.g., Raspberry Pi or NVIDIA Jetson) for processing.
4.  **RF Inference + SHAP Engine**: The edge computer extracts the 141 features, runs the Random Forest model to classify the mechanical state, and computes local SHAP values.
5.  **XAI Dashboard & Safety Controller (PLC)**:
    *   *XAI Dashboard (Visual Branch)*: The SHAP values and diagnostics are sent to a local human-machine interface (HMI) for operators to inspect model decisions.
    *   *Safety Controller (PLC) (Control Branch)*: Model predictions are sent to a safety Programmable Logic Controller (PLC) to trigger warning or emergency stop loops.

---

## 🔄 3. Flowchart 2: Data Flow & Diagnostic Pipeline
Outlines the operational steps of the data processing pipeline:

1.  **Modality Inputs**: Accelerometers, microphones, and current sensors capture the physical states of the motor.
2.  **DAQ Synchronisation**: Merges the multi-modal signals into a single temporally aligned signal block.
3.  **Feature Extraction (43-Dim Vector)**: Computes time-frequency statistical indicators (43 features representing cumulative moments, centroids, and harmonics across vibration, acoustic, and current sensors).
4.  **Random Forest Classifier**: Classifies the feature vector.
5.  **SHAP Explainability (XAI Dashboard)**: Computes feature attributions to provide explainability.
6.  **Control Outputs**:
    *   `Normal` -> **Normal Operation (No Intervention)**.
    *   `Moderate` -> **Reduce Motor Torque (Alignment Mode)**.
    *   `High Severity` -> **Emergency Stop (BPFI / BPFO)**.

---

## 🔀 4. Flowchart 3: Decision & Control Logic
Outlines the safety control loop logic:

1.  **AI Inference Triggered**: Inference runs in real-time.
2.  **Anomaly Detected?**:
    *   `No` -> **Continue Operations**.
    *   `Yes` -> Check **Sensor Integrity OK?** (evaluates for cable breaks or sensor failure).
        *   `No` -> Trigger **Sensor Fault Alarm** (light orange warning).
        *   `Yes` -> Check **Anomaly Persists > 3s?** (filters out transient noise spikes).
            *   `No` -> **Log Transient** in safety files.
            *   `Yes` -> Check **Severity > 80%?** (evaluates the model confidence/probability).
                *   `No` -> **Reduce Torque** (speeds up cooling and shifts the motor out of resonance).
                *   `Yes` -> Trigger **EMERGENCY STOP** (immediate shutdown to prevent catastrophic bearing failure).
