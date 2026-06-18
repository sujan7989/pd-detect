# 🧠 PD Detect — Parkinson's Disease Voice Analysis

> Research & Educational Use Only. Not a medical diagnosis.

A full-stack AI-powered web application for Parkinson's Disease screening through voice biomarker analysis.

## 🚀 Live Demo
- **Frontend:** https://pd-detect.vercel.app
- **API:** https://pd-detect-api.onrender.com
- **API Docs:** https://pd-detect-api.onrender.com/docs

## ✨ Features
- 🎙️ Live voice recording in browser (no app install needed)
- 📂 Upload WAV/MP3/OGG/WebM/FLAC audio files
- 🤖 68+ acoustic features: MFCCs, jitter, shimmer, HNR, pitch, ZCR
- 🧠 3-model ensemble: Random Forest + XGBoost + Calibrated SVM
- 📊 Interactive charts: radar, bar, feature importance
- 📋 Analysis history (saved in browser, persists across sessions)
- 📄 PDF report download
- 📱 Mobile responsive dark UI

## 🛠️ Tech Stack
**Backend:** Python, FastAPI, scikit-learn, XGBoost, numpy, scipy, soundfile, ReportLab  
**Frontend:** React 18, Vite, Tailwind CSS, Recharts, Axios  
**Deployed:** Vercel (frontend) + Render (backend)

## 📊 ML Model
Trained on UCI Parkinson's Dataset (195 samples, 22 features)
- Random Forest: ~94.9% accuracy
- XGBoost: ~94.9% accuracy
- SVM (Calibrated): ~87.2% accuracy
- **Ensemble: ~92.3% accuracy**

## 🖥️ Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
python model_trainer.py      # Train models (once)
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## 📁 Project Structure
```
parkinsons-detection/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── feature_extractor.py # Audio → 68 features (pure numpy/scipy)
│   ├── model_trainer.py     # Train RF+XGBoost+SVM on UCI dataset
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/              # Trained model files (.pkl)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api/client.js    # API + localStorage
│   │   ├── pages/           # Analyze, Dashboard, History, About
│   │   └── components/      # AudioRecorder, ResultPanel, Charts...
│   └── vercel.json
├── render.yaml
└── README.md
```

## ⚠️ Disclaimer
For research and educational purposes only. Not FDA approved. Not a medical diagnosis.
