document.addEventListener('DOMContentLoaded', () => {

    // File Input Logic
    const inputs = ['vib', 'acous', 'tdms'];
    const files = { vib: null, acous: null, tdms: null };
    const btn = document.getElementById('analyzeBtn');

    inputs.forEach(type => {
        const input = document.getElementById(`${type}File`);
        const nameDisp = document.getElementById(`${type}Name`);
        const dropZone = document.getElementById(`${type}Drop`);

        input.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                files[type] = e.target.files[0];
                nameDisp.textContent = files[type].name;
                dropZone.style.borderColor = 'var(--success)';
            } else {
                files[type] = null;
                nameDisp.textContent = '';
                dropZone.style.borderColor = '';
            }
            checkEnableButton();
        });

        // Drag & Drop visual feedback
        dropZone.addEventListener('dragover', () => dropZone.classList.add('dragover'));
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', () => dropZone.classList.remove('dragover'));
    });

    function checkEnableButton() {
        if (files.vib && files.acous && files.tdms) {
            btn.disabled = false;
        } else {
            btn.disabled = true;
        }
    }

    // Chart.js Instances
    let confidenceChartInstance = null;
    let radarChartInstance = null;

    // Form Submission
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        document.getElementById('loading').classList.remove('hidden');
        document.getElementById('resultBanner').classList.add('hidden');
        document.getElementById('probCard').classList.add('hidden');
        document.getElementById('radarCard').classList.add('hidden');
        document.getElementById('xaiCard').classList.add('hidden');
        btn.disabled = true;

        const formData = new FormData();
        formData.append('vib_file', files.vib);
        formData.append('acous_file', files.acous);
        formData.append('tdms_file', files.tdms);
        formData.append('chunk_index', 0); // Always analyzing the 1st window for the demo

        try {
            // Update this URL if hosted externally. For local test, it's localhost:5000
            const response = await fetch('https://multi-sensor-diagnostics-dashboard.onrender.com/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const data = await response.json();
            updateDashboard(data);

        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error(error);
        } finally {
            document.getElementById('loading').classList.add('hidden');
            checkEnableButton();
        }
    });

    const faultDescriptions = {
        'Normal': 'Normal Operation',
        'BPFI': 'Inner Race Fault (BPFI)',
        'BPFO': 'Outer Race Fault (BPFO)',
        'Misalign': 'Shaft Misalignment',
        'Unbalance': 'Rotor Unbalance'
    };

    function updateDashboard(data) {
        // Unhide elements
        document.getElementById('resultBanner').classList.remove('hidden');
        document.getElementById('probCard').classList.remove('hidden');
        document.getElementById('radarCard').classList.remove('hidden');
        document.getElementById('xaiCard').classList.remove('hidden');

        // 1. Update Banner
        const stateEl = document.getElementById('predictedState');
        const bannerEl = document.getElementById('resultBanner');

        const descriptiveState = faultDescriptions[data.prediction] || data.prediction;
        stateEl.textContent = descriptiveState;

        // Remove previous state classes
        bannerEl.className = 'diagnosis-banner glass-panel';
        stateEl.className = 'state-text';

        // Add new state class for coloring (keep original raw class name for CSS hooks)
        bannerEl.classList.add(data.prediction);
        stateEl.classList.add(data.prediction);

        const maxProb = Math.max(...Object.values(data.probabilities));
        document.getElementById('mainConfidence').textContent = `${(maxProb * 100).toFixed(1)}% Confidence`;

        // 2. Update Confidence Chart (Translate labels)
        const descriptiveProbs = {};
        for (const [key, value] of Object.entries(data.probabilities)) {
            descriptiveProbs[faultDescriptions[key] || key] = value;
        }
        updateConfidenceChart(descriptiveProbs);

        // 3. Update Radar Chart (Fingerprint)
        updateRadarChart(data.features, descriptiveState);

        // 4. Update Feature Pills
        updateFeaturePills(data.features);
    }

    // Charting Configuration
    Chart.defaults.color = '#8b9bb4'; // text-secondary
    Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";

    function updateConfidenceChart(probs) {
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        const labels = Object.keys(probs);
        const values = Object.values(probs).map(v => v * 100);

        // Colors: Soft Green for Normal, Soft Red for faults
        const bgColors = labels.map(l => l.includes('Normal') ? 'rgba(52, 211, 153, 0.7)' : 'rgba(251, 113, 133, 0.7)');
        const borderColors = labels.map(l => l.includes('Normal') ? 'rgba(52, 211, 153, 1)' : 'rgba(251, 113, 133, 1)');

        if (confidenceChartInstance) {
            confidenceChartInstance.destroy();
        }

        confidenceChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: values,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(11, 14, 20, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#e2e8f0',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: (ctx) => `${ctx.raw.toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        border: { display: false }
                    },
                    x: {
                        grid: { display: false },
                        border: { display: false }
                    }
                }
            }
        });
    }

    function updateRadarChart(features, prediction) {
        const ctx = document.getElementById('radarChart').getContext('2d');

        // Select key features to show on radar
        const keyFeats = [
            'Vib_RMS', 'Vib_Kurtosis', 'Vib_P2P',
            'Acoustic_RMS', 'Acoustic_Kurtosis', 'Acoustic_P2P',
            'TDMS_Ch0_RMS', 'TDMS_Ch2_RMS'
        ];

        // Friendly names for radar chart
        const friendlyNames = {
            'Vib_RMS': 'Vibration RMS',
            'Vib_Kurtosis': 'Vibration Kurtosis',
            'Vib_P2P': 'Vibration Peak-to-Peak',
            'Acoustic_RMS': 'Acoustic RMS',
            'Acoustic_Kurtosis': 'Acoustic Kurtosis',
            'Acoustic_P2P': 'Acoustic Peak-to-Peak',
            'TDMS_Ch0_RMS': 'Current RMS (Ch 0)',
            'TDMS_Ch2_RMS': 'Current RMS (Ch 2)'
        };

        const displayLabels = keyFeats.map(k => friendlyNames[k] || k.replace(/_/g, ' '));

        // Log transform for visual spread
        const rawValues = keyFeats.map(k => features[k] || 0);
        const normValues = rawValues.map(v => Math.log1p(Math.abs(v)));

        if (radarChartInstance) {
            radarChartInstance.destroy();
        }

        radarChartInstance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: displayLabels,
                datasets: [{
                    label: prediction + ' Profile',
                    data: normValues,
                    backgroundColor: 'rgba(99, 102, 241, 0.15)', // Indigo transparent
                    borderColor: 'rgba(99, 102, 241, 0.8)',
                    pointBackgroundColor: '#0b0e14',
                    pointBorderColor: 'rgba(99, 102, 241, 1)',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(99, 102, 241, 1)',
                    pointBorderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                elements: { line: { tension: 0.4, borderWidth: 2 } },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.04)' },
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        pointLabels: { color: '#8b9bb4', font: { size: 10, family: 'Inter' } },
                        ticks: { display: false, max: Math.max(...normValues) * 1.2 }
                    }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true, color: '#f1f5f9' } },
                    tooltip: {
                        backgroundColor: 'rgba(11, 14, 20, 0.9)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1
                    }
                }
            }
        });
    }

    function updateFeaturePills(features) {
        const container = document.getElementById('featurePills');
        container.innerHTML = ''; // Clear existing

        // Show first 15 features to not overwhelm
        Object.entries(features).slice(0, 15).forEach(([key, value]) => {
            const pill = document.createElement('div');
            pill.className = 'pill';

            const label = document.createElement('span');
            label.textContent = key.replace(/_/g, ' ');

            const val = document.createElement('strong');
            // Format number nicely
            val.textContent = Math.abs(value) < 0.01 && value !== 0 ? value.toExponential(2) : value.toFixed(3);

            pill.appendChild(label);
            pill.appendChild(val);
            container.appendChild(pill);
        });
    }
});
