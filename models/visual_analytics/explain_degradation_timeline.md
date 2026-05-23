# 📊 Technical Specification: Figure 7 - Simulated Health Degradation Timeline

## 1. Visual Specification
*   **Filename**: [health_degradation_timeline.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/health_degradation_timeline.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/health_degradation_timeline.pdf)
*   **Dimensions**: 3300 x 1500 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf`
*   **Aesthetics**: Dual y-axis or shared axis plotting Health Index (blue solid line) and Anomaly Score (red dashed line) over 30 days, with 3 distinct shaded operation zones (green, yellow, red) and vertical orange/red marker boundaries.

## 2. Engineering Context & Methodology
This simulated run-to-failure timeline demonstrates the predictive maintenance capabilities of the diagnostic framework. It shows the transition from normal operation, through early fault development, to critical failure.

The **Health Index** ($HI_t$) and **Anomaly Score** ($AS_t$) are modeled as complementary values over operating time $t$:

$$HI_t = \max\left(0, 100 - AS_t \right)$$

Degradation is modeled as three distinct wear phases:
1.  **Normal Operation (Days 0 to 15)**: The machine operates in a healthy state. The Health Index decays slowly ($100\%$ down to $92.5\%$), and the anomaly score stays low (below $10\%$).
2.  **Incipient Fault Zone (Days 15 to 25)**: An early fault develops (e.g., micro-spall on inner/outer race). The Health Index declines faster (down to $52.5\%$) and the anomaly score rises, representing early warnings.
3.  **Critical Failure Region (Days 25 to 30)**: Severe wear occurs, leading to catastrophic failure. The Health Index drops to $0\%$, and the anomaly score reaches $100\%$.

## 3. Boundary & Trigger Diagnostics
*   **First Anomaly Detected (Day 15)**: A yellow vertical dashed line. Triggered when the anomaly score exceeds a threshold ($AS_t > 12\%$), signaling early wear.
*   **Safe-Stop Triggered (Day 25)**: A red vertical dashed line. Triggered when the anomaly score exceeds a critical safety threshold ($AS_t > 50\%$), prompting the PLC to safely shut down the motor to prevent catastrophic damage.

## 4. Key Takeaways for Paper Reviewers
1.  **Predictive Maintenance Framework**: Demonstrates the practical application of your classification metrics in real-time machine health monitoring and early warning triggers.
2.  **Edge Safety Loop Integration**: Highlights the integration of your model predictions with safety PLC controllers, showing how the diagnostic framework prevents mechanical failure.
