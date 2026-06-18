# 🧠 PD Detect — Parkinson's Disease Voice Analysis

> **Research & Educational Use Only.** Not a medical diagnosis. Always consult a qualified neurologist.

A production-grade, full-stack AI web application for Parkinson's Disease screening through voice biomarker analysis. Built with Python ML backend and React frontend, deployed and accessible to anyone worldwide.

---

## 🌐 Live Application

| Service | URL |
|---|---|
| **Web App** | **https://pd-detect.vercel.app** |
| **API** | https://pd-detect-api.onrender.com |
| **API Docs** | https://pd-detect-api.onrender.com/docs |
| **Source Code** | https://github.com/sujan7989/pd-detect |

> **The client only needs to open: https://pd-detect.vercel.app**
> No installation. No setup. Works in any browser on any device.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ **Live Recording** | Record voice directly in browser — no app install |
| 📂 **File Upload** | Upload WAV, MP3, OGG, WebM, FLAC |
| 🤖 **68+ Acoustic Features** | MFCCs (39), pitch, jitter (5), shimmer (6), HNR, ZCR, spectral features |
| 🧠 **Ensemble ML Model** | Random Forest + XGBoost + SVM trained on UCI data |
| 📊 **Interactive Charts** | Radar chart, feature bar chart, feature importance |
| 🔴 **Confidence Ring** | Animated SVG gauge showing prediction confidence |
| 📋 **Analysis History** | Saved in browser — persists across sessions |
| 📈 **Dashboard** | Statistics and charts from all analyses |
| 📄 **PDF Reports** | Clinical-style downloadable report |
| 🌑 **Dark Premium UI** | Animated particles, glassmorphism design |
| 📱 **Mobile Responsive** | Works on phone and tablet |

---

## 🤖 Machine Learning

### Dataset
UCI Parkinson's Dataset — 195 voice recordings from 31 individuals (23 with PD, 8 healthy)
> Little MA et al. (2007). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.*

### Models Trained
| Model | Accuracy |
|---|---|
| Random Forest (200 trees) | ~94.9% |
| XGBoost (200 estimators) | ~94.9% |
| Calibrated SVM (RBF kernel) | ~87.2% |
| **Ensemble (Voting Classifier)** | **~92.3%** |

### Features Extracted
- **MFCCs**: 13 coefficients + 13 deltas + 13 delta-deltas = 39 features
- **Pitch**: Mean and standard deviation (via autocorrelation)
- **Jitter**: Local, Absolute, RAP, PPQ5, DDP (5 variants)
- **Shimmer**: Local, dB, APQ3, APQ5, APQ11, DDA (6 variants)
- **HNR**: Harmonic-to-Noise Ratio (cepstrum method)
- **ZCR**: Zero Crossing Rate
- **Spectral**: Centroid, Rolloff, Bandwidth
- **Nonlinear**: RPDE proxy, DFA proxy, Spread1, Spread2, PPE

---

## 🛠️ Tech Stack

**Backend (Python)**
- FastAPI — REST API framework
- scikit-learn — Random Forest, SVM, VotingClassifier
- XGBoost — Gradient boosting
- numpy + scipy — Audio feature extraction (no librosa — saves 350MB RAM)
- soundfile — WAV audio reading
- ffmpeg — WebM/MP3/OGG browser recording support
- ReportLab — PDF report generation
- joblib — Model serialization

**Frontend (React)**
- React 18 + Vite — UI framework
- Tailwind CSS v3 — Styling
- Recharts — Data visualizations
- react-router-dom — Navigation
- Axios — HTTP client
- Canvas API — Live waveform visualizer
- localStorage — History persistence

**Deployment**
- Vercel — Frontend (CDN, global)
- Render — Backend Python API (free tier)
- GitHub — Source code

---

## 📁 Project Structure

```
parkinsons-detection/
├── backend/
│   ├── main.py                 # FastAPI — all endpoints
│   ├── feature_extractor.py    # Pure numpy/scipy audio → 68 features
│   ├── model_trainer.py        # Train ML models on UCI dataset
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container for deployment
│   ├── render.yaml             # Render deployment config
│   └── models/                 # Trained model files
│       ├── ensemble.pkl        ✅ included
│       ├── random_forest.pkl   ✅ included
│       ├── xgboost.pkl         ✅ included
│       ├── svm.pkl             ✅ included
│       ├── scaler.pkl          ✅ included
│       └── feature_importance.json
└── frontend/
    ├── src/
    │   ├── App.jsx             # Main app with routing
    │   ├── api/client.js       # API + localStorage history
    │   ├── pages/
    │   │   ├── AnalyzePage.jsx # Main analysis page
    │   │   ├── DashboardPage.jsx
    │   │   ├── HistoryPage.jsx
    │   │   └── AboutPage.jsx
    │   └── components/
    │       ├── AudioRecorder.jsx
    │       ├── WaveformVisualizer.jsx
    │       ├── ResultPanel.jsx
    │       ├── ConfidenceRing.jsx
    │       ├── AcousticCharts.jsx
    │       └── ParticleField.jsx
    ├── vercel.json
    └── package.json
```

---

## 🚀 Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# Models already included — no training needed
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Or use the batch files (Windows)
```
Double-click: START_BACKEND.bat
Double-click: START_FRONTEND.bat
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check + model status |
| POST | `/analyze` | Analyze audio file → prediction |
| POST | `/report` | Generate PDF report |
| GET | `/stats` | Aggregate statistics |

### POST /analyze — Response
```json
{
  "id": "A1B2C3D4",
  "timestamp": "2025-01-15 10:30:00 UTC",
  "prediction": "Healthy",
  "confidence": 72.5,
  "risk_level": "Low",
  "model_used": "Ensemble (RF + XGBoost + SVM)",
  "features": {
    "pitch_mean": 182.3,
    "jitter_local": 0.0038,
    "shimmer_local": 0.0176,
    "hnr": 24.67,
    "mfcc_1_mean": -245.3
  },
  "feature_importance": { ... },
  "recommendations": [ ... ]
}
```

---

## ⚠️ Important Disclaimer

This software is intended **solely for research and educational purposes**. It:

- ❌ Does **not** provide medical diagnoses
- ❌ Is **not** FDA approved or clinically validated
- ❌ Should **not** replace professional medical advice
- ✅ Is based on published academic research (Little et al., 2007)
- ✅ Uses real UCI clinical voice data for training

**Always consult a board-certified neurologist for Parkinson's Disease evaluation.**

---

## 📜 License

MIT License — Free to use for research and educational purposes.
