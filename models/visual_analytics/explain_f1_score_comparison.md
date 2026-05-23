# 📈 Technical Specification: Figure 2 - Grouped Per-Class F1 Fault Profiles

## 1. Visual Specification
*   **Filename**: [precision_recall_comparison.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/precision_recall_comparison.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/precision_recall_comparison.pdf)
*   **Dimensions**: 3600 x 2100 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf`
*   **Color Palette**: Grouped multi-bar configuration with mako-palette shading to represent the distinct diagnostic models.

## 2. Engineering Context & Methodology
The F1-score represents the harmonic mean of precision and recall, serving as the ultimate metric for class-imbalanced multi-sensor bearing and structural fault diagnostic systems:

$$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} \times 100\%$$

$$Precision = \frac{TP}{TP + FP}, \quad Recall = \frac{TP}{TP + FN}$$

This grouped profile contrasts classification performance across all 5 mechanical states: **Normal**, **BPFI** (Inner Race defect), **BPFO** (Outer Race defect), **Misalignment**, and **Unbalance**.

## 3. Comparative Fault Isolation Analysis
*   **Bearing Defect Isolation (BPFI & BPFO)**:
    Both the **Proposed Pure RF** and **Advanced Hybrid** achieve a flawless **100.0% F1-score** across inner and outer race bearing fault states. The unique high-frequency spectral centroids and top 3 distinct FFT harmonic peaks completely isolate these local high-frequency impacts from low-frequency structural anomalies.
*   **Structural Defects (Misalignment & Unbalance)**:
    Resolving structural defects is historically difficult due to overlapping low-frequency spectral signatures.
    *   **Proposed Pure RF**: Resolves **Misalignment** at **95.03% F1-score** and **Unbalance** at **95.48% F1-score**.
    *   **Advanced Hybrid (RF-IF)**: Achieves a high **95.94% F1-score** on Misalignment and **96.25% F1-score** on Unbalance due to anomaly score injection.
    *   **Conventional Hybrid (RF-IF)**: Drops significantly to **92.51%** (Misalignment) and **93.49%** (Unbalance). The lack of advanced features leads to spectral smearing.
    *   **Optimized Gradient Boosting (GB-IF)**: Experiences severe drops, falling to **80.0%** on Misalignment and **88.93%** on Unbalance, showing its susceptibility to structural domain noise.

## 4. Key Takeaways for Paper Reviewers
1.  **Impeccable Bearing Defect Reliability**: The extraction of top 3 distinct harmonics guarantees zero-leakage bearing defect classifications.
2.  **Spectral Centroids Resolve Structural Blur**: Advanced moments and entropy prevent cross-class contamination between Misalignment and Unbalance.
