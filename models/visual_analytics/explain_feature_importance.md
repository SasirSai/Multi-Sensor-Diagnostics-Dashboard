# 📊 Technical Specification: Figure 3 - Proposed Pure RF Gini Feature Importances

## 1. Visual Specification
*   **Filename**: [feature_importance_comparison.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/feature_importance_comparison.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/feature_importance_comparison.pdf)
*   **Dimensions**: 3000 x 1500 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf`
*   **Color Palette**: Modern horizontal crest palette (shades of dark green and blue).

## 2. Engineering Context & Methodology
Feature importance is computed dynamically by loading the Gini impurity decrease values directly from the trained proposed classifier (`models/rf_model.joblib`). 

The Gini importance (Mean Decrease in Impurity, MDI) of a feature $X_j$ is calculated as the sum of all node splits in the ensemble that partition on feature $X_j$, weighted by the fraction of samples reaching those nodes:

$$MDI(X_j) = \frac{1}{N_{trees}} \sum_{t=1}^{N_{trees}} \sum_{i \in \text{splits}(t, X_j)} \frac{N_i}{N_{total}} \left( G(i) - \frac{N_{i, L}}{N_i} G(i, L) - \frac{N_{i, R}}{N_i} G(i, R) \right)$$

where $G(i)$ represents the Gini impurity at split node $i$:

$$G(i) = 1 - \sum_{c=1}^{C} p_c^2$$

## 3. Detailed Gini Importance Attribution
The top 5 features plotted represent the core information drivers of the multi-sensor pipeline:
1.  **TDMS Ch0 Mean (Current Phase A Mean)**: Attributes the primary baseline voltage/current offset. Essential for detecting magnetic field asymmetry under Unbalance.
2.  **TDMS Ch0 RMS (Current Phase A RMS)**: Reflects total current power. Captures the severe load oscillations and power fluctuations caused by structural defects.
3.  **TDMS Ch0 Spectral Energy**: Attributes current frequency-domain magnitude, isolating magnetic flux modulations caused by mechanical fault frequencies.
4.  **Vibration Spectral Energy**: Attributes cumulative mechanical energy across acoustic/vibration bands, identifying high-frequency friction and impact spikes.
5.  **TDMS Ch1 RMS (Current Phase B RMS)**: Demonstrates the importance of multi-phase current analysis in isolating directional misalignment.

## 4. Key Takeaways for Paper Reviewers
1.  **Electro-Mechanical Fusion Power**: 4 of the top 5 features are derived from electromagnetic current telemetry (TDMS channels), proving that Current Signature Analysis (MCSA) is highly sensitive to mechanical defects.
2.  **Gini Objectivity**: The chart reflects physical model splits, providing empirical proof of sensor modalities contribution without manual heuristic bias.
