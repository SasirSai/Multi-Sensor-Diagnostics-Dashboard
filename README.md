# ⚙️ Machine Diagnostics Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" />
</p>

A subtle, premium web dashboard that provides real-time predictive analytics and explainable AI for multi-sensor fault diagnosis in rotating machines.

This project utilizes a **FastAPI** backend to parse heavy engineering data files (`.mat`, `.tdms`), extract statistical time-domain features, and run a pre-trained **Random Forest** model. The predictions and telemetry are then served to a lightweight, highly-styled **Vanilla HTML/CSS/JS** frontend dashboard for real-time visualization.

---

## ✨ Key Features

- **Multi-Sensor Fusion**: Supports simultaneous ingestion of Vibration (`.mat`), Acoustic (`.mat`), and Current/Temperature (`.tdms`) signals.
- **Real-Time Predictive Analytics**: Automatically classifies the machine state based on 43 extracted statistical features.
- **Explainable AI (XAI) Dashboard**: Visualizes the model's confidence distribution and provides a radar chart of the sensor feature profile (Fingerprint).
- **Premium User Interface**: A modern, glassmorphism-inspired UI with smooth micro-interactions.
- **Lightweight Backend**: Built on FastAPI for lightning-fast inference and data processing.

---

## 📁 Project Structure

```text
📦 Machine Diagnostics Engine
 ┣ 📂 backend/         # FastAPI server (main.py) and dependencies
 ┣ 📂 frontend/        # Web application (index.html, app.js, styles.css)
 ┣ 📂 models/          # Trained ML weights (.joblib files)
 ┣ 📂 Dataset/         # Raw telemetry data (.mat, .tdms)
 ┣ 📂 temp_uploads/    # Temporary directory for uploaded files during inference
 ┗ 📜 export_model.py  # Script to train and export the Random Forest model
```

---

## 🚀 How to Run the Project Locally

Follow these steps to set up the environment, train the artificial intelligence model, start the server, and launch the dashboard.

### Step 1: Install Python Requirements

Open a terminal in the root directory of the project and install all the necessary Python libraries for data processing and the backend server:

```bash
pip install -r backend/requirements.txt
```

### Step 2: Extract Features & Train the Model

Before the dashboard can make predictions, the AI needs to be trained on the `Dataset/` folder. This script will scan the folder, extract statistical features from the signals, train a Random Forest model, and save it into the `models/` directory.

```bash
python export_model.py
```
> *(Wait for the console to output "Model and metadata successfully exported to models" before proceeding)*

### Step 3: Start the FastAPI Backend Server

The backend acts as the bridge between the web browser and the Python ML logic. Start the server by running:

```bash
python backend/main.py
```
> *(Leave this terminal window open! The server must continuously run in the background to receive requests. You should see `Application startup complete` in the console.)*

### Step 4: Open the Web Dashboard

The frontend requires no compilation or build steps. You can open it directly in any modern web browser!

1. Open your File Explorer.
2. Navigate to the `frontend/` directory.
3. Double-click `index.html` (it will open securely in your default browser).

### Step 5: Analyze Telemetry 

1. On the dashboard, drag and drop test sensor files into their respective uploading zones. For example:
   - **Vibration:** Drag `Dataset/vibration/0Nm_BPFI_03.mat`
   - **Acoustic:** Drag `Dataset/acoustic/0Nm_BPFI_03.mat`
   - **Current & Temp:** Drag `Dataset/current,temp/0Nm_BPFI_03.tdms`
2. Click the **Analyze Telemetry** button.
3. The dashboard will instantly compute the results, visualize the machine state, probability distribution, and the unique sensor fingerprint!

---

## 🌐 Deployment Guidelines

If you wish to host this dashboard publicly on the internet:
1. **Backend**: Deploy the `backend` folder as a web service via platforms like **Render**, **Railway**, or **Heroku**.
2. **API Configuration**: Update the `fetch(URL)` call inside `frontend/app.js` to point to your new cloud API address instead of `localhost:5000`.
3. **Frontend**: Deploy the `frontend` folder securely and for free on **GitHub Pages**, **Vercel**, or **Netlify**.

---

<p align="center">
  <i>Diagnosing the Future of Rotating Machines 🚀</i>
</p>
