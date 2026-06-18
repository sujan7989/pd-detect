# 🧠 PD Detect — Parkinson's Disease Voice Analysis Platform

> **Research & Educational Use Only.** Not a medical diagnosis tool.
> Always consult a qualified neurologist for medical evaluation.

A production-grade, full-stack web application for Parkinson's Disease screening
through acoustic voice biomarker analysis. Uses a trained ensemble ML model (Random Forest
+ XGBoost + SVM) on the UCI Parkinson's Dataset with a premium dark-mode UI.

---

## ✨ Live Demo

| Service  | URL                         |
|----------|-----------------------------|
| Frontend | http://localhost:5173       |
| Backend  | http://localhost:8000       |
| API Docs | http://localhost:8000/docs  |

---

## 🚀 Quick Start (One-Click)

Double-click these `.bat` files in the project root:

1. **`START_BACKEND.bat`** — Installs dependencies & starts FastAPI server
2. **`START_FRONTEND.bat`** — Installs dependencies & starts React dev server
3. **`TRAIN_MODEL.bat`** — Downloads UCI dataset & trains the ML ensemble (run once)

Then open **http://localhost:5173** in your browser.

---

## 📋 Manual Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Internet connection (first run only, for dataset download)

### Backend
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Train the ML model (required once — downloads UCI dataset ~3KB, takes ~2 min)
python model_trainer.py

# Start the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🏗 Project Structure

```
parkinsons-detection/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── model_trainer.py         # ML training script
│   ├── feature_extractor.py     # Audio → 60+ acoustic features
│   ├── requirements.txt
│   └── models/                  # Saved models (after training)
│       ├── ensemble.pkl
│       ├── scaler.pkl
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       ├── svm.pkl
│       └── feature_importance.json
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── api/
        │   └── client.js          # Axios API client
        ├── pages/
        │   ├── AnalyzePage.jsx    # Main analysis page
        │   ├── DashboardPage.jsx  # Analytics dashboard
        │   ├── HistoryPage.jsx    # Analysis history
        │   └── AboutPage.jsx      # Documentation
        └── components/
            ├── AudioRecorder.jsx  # Record/upload + analyze
            ├── WaveformVisualizer.jsx  # Live canvas waveform
            ├── ResultPanel.jsx    # Full results display
            ├── ConfidenceRing.jsx # Animated SVG confidence gauge
            ├── AcousticCharts.jsx # Radar + bar + importance charts
            └── ParticleField.jsx  # Background particle animation
```

---

## 🤖 ML Model

| Model          | Accuracy | Notes                        |
|----------------|----------|------------------------------|
| Random Forest  | ~94.9%   | 200 trees                    |
| XGBoost        | ~94.9%   | 200 estimators, lr=0.05      |
| SVM (Calib.)   | ~87.2%   | RBF kernel, C=10             |
| **Ensemble**   | **~92.3%** | Soft-voting, 5-fold CV    |

Trained on UCI Parkinson's Dataset (195 samples, 22 features).
*Little MA et al. (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.*

---

## 🔌 API Reference

| Method | Endpoint        | Description                     |
|--------|-----------------|---------------------------------|
| GET    | `/health`       | Health check + model status     |
| POST   | `/analyze`      | Analyze audio file              |
| GET    | `/history`      | Get all past analyses           |
| DELETE | `/history`      | Clear all history               |
| DELETE | `/history/{id}` | Delete single record            |
| GET    | `/stats`        | Aggregate dashboard statistics  |
| POST   | `/report`       | Generate PDF report             |

### POST /analyze — Example Response
```json
{
  "id": "A1B2C3D4",
  "timestamp": "2025-01-15 10:30:00 UTC",
  "prediction": "Parkinson's Detected",
  "confidence": 87.3,
  "probability": 0.873,
  "risk_level": "High",
  "features": {
    "jitter_local": 0.00892,
    "shimmer_local": 0.07321,
    "hnr": 12.45,
    "pitch_mean": 142.3,
    "mfcc_1_mean": -245.3
  },
  "feature_importance": {
    "PPE": { "importance": 0.168, "value": 0.234, "normal_min": null, "normal_max": null }
  },
  "recommendations": ["Consult a neurologist..."],
  "model_used": "Ensemble (RF + XGBoost + SVM)"
}
```

---

## 🎨 Tech Stack

**Backend:** FastAPI · librosa · scikit-learn · XGBoost · ReportLab · soundfile · numpy · pandas

**Frontend:** React 18 · Vite 5 · Tailwind CSS v3 · Recharts · react-router-dom ·
react-dropzone · react-hot-toast · lucide-react · axios

**ML:** Random Forest · XGBoost · Calibrated SVM · VotingClassifier · StandardScaler

---

## ⚠️ Disclaimer

This software is for **research and educational purposes only**. It:
- Does **not** provide medical diagnoses
- Is **not** FDA approved or clinically validated
- Should **not** replace professional medical advice
- Is based on a dataset of only 195 samples

**Always consult a board-certified neurologist for Parkinson's Disease evaluation.**
