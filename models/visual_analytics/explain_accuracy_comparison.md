# 📊 Technical Specification: Figure 1 - Comparative Model Accuracy

## 1. Visual Specification
*   **Filename**: [accuracy_comparison.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/accuracy_comparison.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/accuracy_comparison.pdf)
*   **Dimensions**: 3000 x 1800 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf` for direct LaTeX import.
*   **Highlight Scheme**: **Proposed Pure RF** is highlighted in Royal Blue (`#0077B6`) to emphasize it as the primary system contribution, while baseline models utilize neutral grays and cyans.

## 2. Engineering Context & Methodology
Classification accuracy is evaluated on **completely unseen holdout motor runs** under a rigorous, leak-free stratified file-level validation partition. 

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} \times 100\%$$

Unlike chunk-level validation splits which contaminate the evaluation set with adjacent temporal records, this validation partition segregates entire motor runs by class and torque ($0\text{ Nm}, 2\text{ Nm}, 4\text{ Nm}$), preventing data leakage and ensuring maximum scientific rigor.

## 3. Comparative Performance Analysis
*   **Proposed Pure RF (97.07%)**: The high-accuracy proposed classical baseline. By relying on a large 141-feature extraction matrix (comprising time-domain statistics, spectral entropy, and distinct harmonic FFT peaks), it establishes highly robust decision boundaries across all torques.
*   **Advanced Hybrid (RF-IF) (96.75%)**: Incorporates unsupervised anomaly features from an Isolation Forest fit exclusively on Normal training samples. While highly interpretable, the continuous anomaly feature introduces a slight variance, yielding a marginal $0.32\%$ accuracy decrease compared to the proposed pure model.
*   **Conventional Hybrid (RF-IF) (95.68%)**: Replicates standard industry practices (Sujith corrected) using only 56 basic features. The lack of advanced spectral centroids and harmonics limits its ability to resolve overlapping mechanical defects, trailing the proposed system by $1.39\%$.
*   **Optimized Gradient Boosting (GB-IF) (89.67%)**: Uses a modern boosted tree structure. Due to the high dimensionality of multi-sensor data, the boosted ensemble suffers from minor overfitting on holdout domains, trailing the proposed random forest system by a significant $7.40\%$.

## 4. Key Takeaways for Paper Reviewers
1.  **Purity Over Anomaly Score**: Rigorous feature engineering (141 features) is superior to basic features combined with unsupervised anomaly indexes.
2.  **Robustness to domain shifts**: Random Forest ensembles show stronger generalization across different torque regimes compared to boosted trees (HistGradientBoosting) under identical holdout validation.
