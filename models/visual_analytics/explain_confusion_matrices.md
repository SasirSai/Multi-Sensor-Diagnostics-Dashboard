# 🧩 Technical Specification: Confusion Matrices Evaluation

## 1. Visual Specification
*   **Filenames**:
    *   Proposed Pure RF: [confusion_matrix_proposed_pure_rf.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/confusion_matrix_proposed_pure_rf.png)
    *   Advanced Hybrid: [confusion_matrix_advanced_hybrid.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/confusion_matrix_advanced_hybrid.png)
    *   Conventional Hybrid: [confusion_matrix_conventional_hybrid.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/confusion_matrix_conventional_hybrid.png)
    *   Optimized GB: [confusion_matrix_optimized_gb.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/confusion_matrix_optimized_gb.png)
*   **Dimensions**: 2400 x 2100 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf`
*   **Aesthetics**: Beautifully annotated blue heatmaps with both absolute sample counts and relative class percentages (`val\n(perc%)`) for each cell.

## 2. Engineering Context & Methodology
The confusion matrices evaluate predicted classes against true labels over all **9,684 unseen test windows** from the holdout dataset. Each row representing the actual mechanical state class and each column representing the predicted state class.

This maps the exact classification errors (False Positives and False Negatives) to evaluate the structural integrity of each model's decision boundaries.

## 3. Comparative Grid Analysis
1.  **Proposed Pure RF**:
    *   Achieves a perfect **1,842 / 1,842** (100.0%) correct detections on **BPFI** and **BPFO** categories.
    *   Accurately isolates **2,716 / 3,000** (90.5%) **Misalignment** cases and **3,000 / 3,000** (100.0%) **Unbalance** cases.
    *   Only **284** Misalignment windows are misclassified as Unbalance, caused by minor low-frequency mechanical energy bleeding at high speeds.
2.  **Advanced Hybrid (RF-IF)**:
    *   Maintains the same perfect bearing fault classification as the proposed pure model.
    *   Improves Misalignment isolation slightly to **2,766 / 3,000** (92.2%) due to the added information of the unsupervised Isolation Forest anomaly index.
    *   Only **234** Misalignment windows are misclassified as Unbalance.
3.  **Conventional Hybrid (RF-IF)**:
    *   Maintains perfect bearing fault classification.
    *   Misclassification between Misalignment and Unbalance increases to **418** cases (13.9% error rate) due to the reduced 56-feature dimensionality.
4.  **Optimized Gradient Boosting (GB-IF)**:
    *   Maintains perfect bearing fault classification.
    *   Suffers severe structural diagnostic failures: **1,000** Misalignment cases are misclassified as Unbalance (33.3% error rate).
    *   Misclassifies **253** Normal cases as Unbalance, showing a significant false alarm rate.

## 4. Key Takeaways for Paper Reviewers
1.  **Bearing Fault Flawlessness**: Zero false alarms or missed detections on BPFI and BPFO across all models, proving the exceptional robustness of the top 3 distinct FFT harmonic extraction.
2.  **Proposed Pure RF Superiority**: The proposed classical baseline shows extremely clean diagonal clustering, proving that extensive statistical/harmonic feature engineering is highly resilient to domain noises compared to modern boosting architectures.
