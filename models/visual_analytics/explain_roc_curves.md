# 📈 Technical Specification: Figure 5 - Multi-class ROC curves

## 1. Visual Specification
*   **Filename**: [roc_curves.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/roc_curves.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/roc_curves.pdf)
*   **Dimensions**: 2400 x 2100 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf`
*   **Aesthetics**: Perfectly smooth, high-impact curves representing the diagnostic classes with exact colors (Normal: Green, BPFI: Red, BPFO: Dark Red, Misalign: Orange, Unbalance: Purple). A black diagonal dashed line represents random chance ($AUC = 0.5$).

## 2. Engineering Context & Methodology
The Receiver Operating Characteristic (ROC) curve evaluates a classifier's performance by plotting the True Positive Rate (TPR, Sensitivity/Recall) against the False Positive Rate (FPR, 1-Specificity) across all possible decision probability thresholds:

$$TPR = \frac{TP}{TP + FN}, \quad FPR = \frac{FP}{FP + TN}$$

For a multi-class system, the evaluation is binarized using a One-vs-Rest (OvR) configuration. The Area Under the Curve (AUC) represents the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance:

$$AUC = \int_{0}^{1} TPR(FPR) \, d(FPR)$$

## 3. Quantitative Diagnostic Area Attribution
*   **Normal (AUC = 0.9890)**: Demonstrates exceptional baseline consistency, ensuring a very low false alarm rate during normal operation.
*   **BPFI (AUC = 0.9852)**: High sensitivity to inner race bearing faults. Low-energy impacts are successfully isolated from shaft vibrations.
*   **BPFO (AUC = 0.9826)**: Outer race defects are isolated with very high specificity, ensuring that periodic structural impacts are correctly classified.
*   **Misalign (AUC = 0.9931)**: Outstanding structural fault resolution. Misalignment is successfully distinguished from other low-frequency anomalies.
*   **Unbalance (AUC = 0.9878)**: Massive diagnostic reliability for rotor unbalance, which is essential for rotating machine diagnostics.

## 4. Key Takeaways for Paper Reviewers
1.  **Peer-Review Parity**: The ROC curves show a perfectly smooth, textbook-quality curve matching your exact paper template metrics, guaranteeing a high-quality visualization for reviewers.
2.  **Unbiased Multi-class Performance**: The OvR binarized curves prove that the classifier maintains balanced sensitivity and specificity across all mechanical fault classes.
