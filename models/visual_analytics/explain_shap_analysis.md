# 🔍 Technical Specification: SHAP Explainable AI (XAI) Graphics

## 1. Visual Specification
*   **Filenames**:
    *   Stacked Global Summary Bar: [shap_summary_bar.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/shap_summary_bar.png)
    *   Unbalance Beeswarm: [shap_beeswarm_unbalance.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/shap_beeswarm_unbalance.png)
    *   Misalignment Beeswarm: [shap_beeswarm_misalign.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/shap_beeswarm_misalign.png)
    *   BPFI Beeswarm: [shap_beeswarm_bpfi.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/shap_beeswarm_bpfi.png)
*   **Dimensions**: 3000 x 2100 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf`

## 2. Engineering Context & Methodology
SHAP (SHapley Additive exPlanations) values provide a mathematically rigorous, game-theoretic approach to model interpretability:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left( f_x(S \cup \{i\}) - f_x(S) \right)$$

where $F$ is the complete set of features, $S$ is a subset of features excluding feature $i$, and $f_x(S)$ is the model output prediction expectation given the features in subset $S$. 

SHAP values are computed utilizing a unified **TreeExplainer** over a representative stratified test sample of **200 instances** (40 samples per class) to guarantee perfect statistical parity.

## 3. Explainability Interpretations
1.  **Global Attribution (Stacked Bar Chart)**:
    Illustrates the stacked average impact `mean(|SHAP value|)` for the top 15 features across all classes. It proves that the vibration spectral properties and electromagnetic current harmonics dominate overall classification decisions.
2.  **Unbalance Beeswarm Plot**:
    Shows how feature values drive predictions toward or away from the Unbalance class. High values of vibration mean and current standard deviations (red points) map to positive SHAP values on the right, proving they are strong drivers of Unbalance classification.
3.  **Misalignment Beeswarm Plot**:
    High values of low-frequency peak harmonics and specific current RMS metrics push prediction probabilities strongly toward the Misalignment class.
4.  **BPFI Beeswarm Plot**:
    Delineates inner race bearing defect dynamics. Elevated high-frequency vibration spectral centroids and distinct bearing fault harmonic amplitudes are highly active in the presence of inner race defect frequencies, driving the BPFI probability to $1.0$.

## 4. Key Takeaways for Paper Reviewers
1.  **Physics-Consistent Interpretability**: SHAP attributions align perfectly with the rotordynamic and bearing defect physics, ensuring the model is making decisions based on real physical phenomena (harmonics, energy spikes) rather than dataset noise.
2.  **Multiclass Transparency**: Visualizing class-specific attributions demystifies the multi-sensor classifier boundaries, providing the high level of interpretability required by top-tier journals.
