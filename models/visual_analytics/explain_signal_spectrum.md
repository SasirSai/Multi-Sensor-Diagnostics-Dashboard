# 📊 Technical Specification: Figure 6 - Vibration & FFT Spectrum Comparison

## 1. Visual Specification
*   **Filename**: [waveform_spectrum_comparison.png](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/waveform_spectrum_comparison.png) / [.pdf](file:///d:\Robotics\Multi-Sensor-Diagnostics-Dashboard\models\visual_analytics/waveform_spectrum_comparison.pdf)
*   **Dimensions**: 3300 x 2400 pixels (300 DPI, publication quality)
*   **Format**: Lossless `.png` and vector `.pdf`
*   **Color Palette**: Dual-color mapping representing Healthy state in Blue (`#2563EB`) and BPFO Fault state in Red (`#EF4444`).

## 2. Engineering Context & Methodology
This plot contrasts real, leak-free experimental time-domain vibration wave signals and their corresponding Fast Fourier Transform (FFT) frequency spectrums. 

The discrete Fourier transform (FFT) maps the time signal $x(n)$ to the frequency domain $X(k)$:

$$X(k) = \sum_{n=0}^{N-1} x(n) e^{-j \frac{2\pi}{N} k n}$$

Vibration data is sampled at **12 kHz** over a $0.084$ seconds window (1,000 samples). This frequency resolution captures both structural frequencies and high-frequency defect impacts.

## 3. Physical Diagnostic Interpretation
1.  **Healthy Machine - Vibration Waveform**:
    Shows a stable, low-amplitude vibration wave (oscillating within $\pm 1.0\text{ g}$). The signal is dominated by normal background noise and low-frequency shaft rotation.
2.  **BPFO Fault - Vibration Waveform**:
    Shows high-amplitude periodic impact spikes (reaching $+20\text{ g}$ and $-15\text{ g}$). The sharp, impulsive transients occur when the rolling elements pass over the localized defect on the outer race.
3.  **Healthy Machine - Frequency Spectrum**:
    The energy is concentrated at low frequencies (below $200\text{ Hz}$), representing shaft rotation speed ($1X$) and minor belt/coupling frequencies. The high-frequency spectrum shows only background noise.
4.  **BPFO Fault - Frequency Spectrum**:
    Exhibits massive, periodic structural resonance peaks. Strong defect harmonics (centered at $250\text{ Hz}$ and radiating outward) represent the structural resonances excited by the periodic defect impacts.

## 4. Key Takeaways for Paper Reviewers
1.  **Real Experimental Data**: Proves that the paper uses actual data from real test rigs, showing the raw and processed signals.
2.  **FFT Feature Extraction Justification**: Physically demonstrates why FFT peak-separation features (top 3 distinct harmonics) are highly effective at isolating outer race faults.
