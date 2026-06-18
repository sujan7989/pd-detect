"""
PD Detect — FastAPI Backend
Endpoints: /health, /analyze, /history, /history/{id}, /stats, /report
"""

import os
import io
import json
import uuid
import random
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from feature_extractor import extract_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PD Detect API",
    description="Parkinson's Disease Detection via Voice Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins — frontend hosted on Vercel/Netlify
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── UCI feature names (22 features, matches training data column order) ─────
UCI_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]

# ─── Load models ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
models_loaded = False
scaler        = None
ensemble      = None
feature_importance_global: dict = {}


def try_load_models():
    global models_loaded, scaler, ensemble, feature_importance_global
    try:
        scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        ensemble = joblib.load(os.path.join(MODELS_DIR, "ensemble.pkl"))
        fi_path  = os.path.join(MODELS_DIR, "feature_importance.json")
        if os.path.exists(fi_path):
            with open(fi_path) as fh:
                feature_importance_global = json.load(fh)
        models_loaded = True
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.warning(f"⚠️  Models not found ({e}). Using heuristic fallback.")
        models_loaded = False


try_load_models()

# ─── In-memory history (max 200 entries) ─────────────────────────────────────
analysis_history: List[dict] = []
MAX_HISTORY = 200

# ─── Normal ranges (adjusted for browser WebM recording characteristics) ──────
NORMAL_RANGES = {
    "jitter_local":           (0.001, 0.08),   # browser jitter is ~8x clinical
    "shimmer_local":          (0.010, 0.12),   # browser shimmer is ~3x clinical
    "hnr":                    (5.0,   35.0),   # browser HNR is ~8dB lower
    "pitch_mean":             (75.0,  300.0),
    "pitch_std":              (0.5,   20.0),
    "zcr_mean":               (0.01,  0.25),
    "spectral_centroid_mean": (300.0, 5000.0),
    "rms_mean":               (0.005, 0.50),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def map_to_uci_vector(features: dict) -> np.ndarray:
    """
    Map extracted audio features to the 22-column UCI feature vector.

    IMPORTANT: Browser WebM recordings have codec compression artifacts
    that inflate jitter/shimmer values vs clinical MDVP measurements.
    We apply a normalization factor to compensate.

    UCI healthy ranges:
      Jitter%: 0.18–1.36%  (mean 0.39%)
      Shimmer: 0.95–4.09%  (mean 1.76%)
      HNR:     17.9–33.0   (mean 24.7 dB)
    """
    # ── Jitter: browser codec inflates this by ~4–8x vs clinical
    # Browser jitter_local is typically 0.02–0.08 for healthy voice
    # UCI clinical healthy jitter% is 0.18–1.36%
    # Correction factor: divide by 8 to map browser → clinical range
    raw_jitter  = features.get("jitter_local", 0.004)
    jitter_adj  = raw_jitter / 8.0   # normalize to clinical scale
    jitter_pct  = jitter_adj * 100.0  # convert to percentage for UCI

    # ── Shimmer: similar inflation from codec, divide by 3
    raw_shimmer = features.get("shimmer_local", 0.025)
    shimmer_adj = raw_shimmer / 3.0

    # ── HNR: browser recordings typically give lower HNR than clinical
    # Add a browser compensation offset of +8 dB
    raw_hnr     = features.get("hnr", 15.0)
    hnr_adj     = float(np.clip(raw_hnr + 8.0, 0.1, 40.0))
    nhr         = 1.0 / (hnr_adj + 1e-6)

    # ── Pitch: use directly, already in correct range
    pitch = float(np.clip(features.get("pitch_mean", 150.0), 50.0, 350.0))

    # ── Other features
    jitter_abs  = features.get("jitter_absolute", 0.00003)
    rpde   = float(np.clip(features.get("rpde",    0.50), 0.0, 1.0))
    dfa    = float(np.clip(features.get("dfa",     0.70), 0.5, 0.9))
    spread1= float(np.clip(features.get("spread1", -5.5), -8.0, -2.0))
    spread2= float(np.clip(features.get("spread2",  0.20), 0.0, 0.5))
    ppe    = float(np.clip(features.get("ppe",      0.20), 0.0, 0.6))
    d2     = float(np.clip(features.get("zcr_mean", 0.05) * 40.0, 1.5, 4.0))

    vec = [
        pitch,                                              # MDVP:Fo(Hz)
        pitch * 1.08,                                       # MDVP:Fhi(Hz)
        pitch * 0.92,                                       # MDVP:Flo(Hz)
        jitter_pct,                                         # MDVP:Jitter(%)
        jitter_abs,                                         # MDVP:Jitter(Abs)
        features.get("jitter_rap",  jitter_pct * 0.0065),  # MDVP:RAP
        features.get("jitter_ppq5", jitter_pct * 0.0077),  # MDVP:PPQ
        features.get("jitter_ddp",  jitter_pct * 0.0196),  # Jitter:DDP
        shimmer_adj,                                        # MDVP:Shimmer
        features.get("shimmer_db",    0.270),               # MDVP:Shimmer(dB)
        features.get("shimmer_apq3",  0.016),               # Shimmer:APQ3
        features.get("shimmer_apq5",  0.019),               # Shimmer:APQ5
        features.get("shimmer_apq11", 0.027),               # MDVP:APQ
        features.get("shimmer_dda",   0.047),               # Shimmer:DDA
        nhr,                                                # NHR
        hnr_adj,                                            # HNR
        rpde,                                               # RPDE
        dfa,                                                # DFA
        spread1,                                            # spread1
        spread2,                                            # spread2
        d2,                                                 # D2
        ppe,                                                # PPE
    ]
    return np.array(vec, dtype=np.float64)


def heuristic_predict(features: dict) -> dict:
    """
    Research-backed heuristic when trained models are unavailable.
    Thresholds derived from Little et al. (2007) and published PD research.
    """
    j   = features.get("jitter_local", 0.005)
    sh  = features.get("shimmer_local", 0.03)
    hnr = features.get("hnr", 20.0)
    ps  = features.get("pitch_std", 2.0)
    dfa = features.get("dfa", 0.65)
    rpde= features.get("rpde", 0.50)

    score = 0.0
    # Jitter thresholds (PD typically > 1% = 0.01 local)
    if j > 0.010:  score += 0.35
    elif j > 0.007: score += 0.15
    elif j < 0.002: score -= 0.20
    # Shimmer thresholds (PD typically > 0.07)
    if sh > 0.070:  score += 0.30
    elif sh > 0.050: score += 0.10
    elif sh < 0.015: score -= 0.20
    # HNR (PD typically < 15 dB)
    if hnr < 12.0:  score += 0.30
    elif hnr < 18.0: score += 0.10
    elif hnr > 25.0: score -= 0.20
    # Pitch variation
    if ps > 8.0:   score += 0.15
    elif ps > 5.0: score += 0.05
    elif ps < 1.5: score -= 0.10
    # DFA / RPDE (nonlinear)
    if dfa  > 0.82: score += 0.10
    if rpde > 0.60: score += 0.08

    # Small Gaussian noise to avoid exactly 50%
    score += random.gauss(0, 0.04)
    prob  = float(np.clip(1.0 / (1.0 + np.exp(-score * 3.5)), 0.05, 0.95))
    prediction = "Parkinson's Detected" if prob >= 0.5 else "Healthy"
    confidence = round((prob if prob >= 0.5 else 1.0 - prob) * 100, 1)
    return {"prediction": prediction, "probability": prob, "confidence": confidence,
            "model_used": "heuristic_screening"}


def risk_level(prob: float) -> str:
    if prob >= 0.70: return "High"
    if prob >= 0.40: return "Medium"
    return "Low"


def get_recommendations(prediction: str, confidence: float, features: dict) -> list:
    recs = []
    if "Parkinson" in prediction:
        recs.append("Consult a board-certified neurologist for a comprehensive evaluation.")
        recs.append("A standardized UPDRS (Unified Parkinson's Disease Rating Scale) assessment is recommended.")
        if features.get("jitter_local", 0) > 0.008:
            recs.append("Elevated pitch irregularity (jitter) detected — a speech-language pathologist can assess vocal symptoms.")
        if features.get("shimmer_local", 0) > 0.05:
            recs.append("High amplitude irregularity (shimmer) noted — voice therapy may help manage vocal fatigue.")
        if features.get("hnr", 20) < 15:
            recs.append("Reduced harmonic-to-noise ratio suggests increased breathiness — further respiratory assessment may be beneficial.")
        recs.append("Maintain a symptom diary to track voice and motor changes over time for clinical appointments.")
        recs.append("Regular aerobic exercise (e.g., cycling, swimming) has demonstrated benefits for motor symptoms in PD.")
        recs.append("LSVT LOUD speech therapy is specifically designed for Parkinson's-related vocal changes.")
        if confidence < 70:
            recs.append("Moderate model confidence — retesting with a quieter environment and longer sustained vowel may improve accuracy.")
    else:
        recs.append("Voice analysis appears within typical acoustic parameters.")
        recs.append("Continue routine health screenings with your primary care physician.")
        recs.append("Maintain vocal health through adequate hydration and avoiding vocal strain.")
        if confidence < 70:
            recs.append("Confidence is moderate — consider retesting with a cleaner recording for a more definitive result.")
        recs.append("This screening does not rule out Parkinson's Disease. Consult a neurologist if you have motor or other concerns.")
    return recs


def build_feature_importance(features: dict, n: int = 10) -> dict:
    """Return top-N feature importances with patient values."""
    UCI_TO_FEATURE = {
        "MDVP:Fo(Hz)":       "pitch_mean",
        "MDVP:Jitter(%)":    "jitter_local",
        "MDVP:Jitter(Abs)":  "jitter_absolute",
        "MDVP:RAP":          "jitter_rap",
        "MDVP:PPQ":          "jitter_ppq5",
        "Jitter:DDP":        "jitter_ddp",
        "MDVP:Shimmer":      "shimmer_local",
        "MDVP:Shimmer(dB)":  "shimmer_db",
        "HNR":               "hnr",
        "NHR":               "hnr",
        "RPDE":              "rpde",
        "DFA":               "dfa",
        "spread1":           "spread1",
        "spread2":           "spread2",
        "PPE":               "ppe",
    }
    if feature_importance_global:
        result = {}
        for uci_name in list(feature_importance_global.keys())[:n]:
            fk   = UCI_TO_FEATURE.get(uci_name, uci_name)
            val  = features.get(fk, 0.0)
            rng  = NORMAL_RANGES.get(fk, (None, None))
            result[uci_name] = {
                "importance": round(feature_importance_global[uci_name], 6),
                "value":      round(float(val), 6),
                "normal_min": rng[0],
                "normal_max": rng[1],
            }
        return result
    # Fallback
    fallback = {
        "Jitter (Local)":  features.get("jitter_local", 0),
        "Shimmer (Local)": features.get("shimmer_local", 0),
        "HNR":             features.get("hnr", 0),
        "Pitch Mean":      features.get("pitch_mean", 0),
        "RPDE":            features.get("rpde", 0),
        "DFA":             features.get("dfa", 0),
        "ZCR Mean":        features.get("zcr_mean", 0),
        "Spread1":         features.get("spread1", 0),
        "Spread2":         features.get("spread2", 0),
        "PPE":             features.get("ppe", 0),
    }
    mx = max(abs(v) for v in fallback.values()) + 1e-10
    return {
        k: {"importance": round(abs(v) / mx, 6), "value": round(float(v), 6),
            "normal_min": None, "normal_max": None}
        for k, v in fallback.items()
    }


def build_pdf(result: dict) -> bytes:
    """Generate a clinical-style PDF report using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=20*mm, leftMargin=20*mm,
                             topMargin=20*mm,  bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    BLUE   = colors.HexColor("#1d4ed8")
    DARK   = colors.HexColor("#0f172a")
    GREY   = colors.HexColor("#475569")
    LIGHT  = colors.HexColor("#f8fafc")
    RED    = colors.HexColor("#dc2626")
    GREEN  = colors.HexColor("#16a34a")

    title_s = ParagraphStyle("T",  parent=styles["Title"],  fontSize=22, textColor=BLUE, alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ParagraphStyle("S",  parent=styles["Normal"], fontSize=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=2)
    h2_s    = ParagraphStyle("H2", parent=styles["Normal"], fontSize=12, textColor=BLUE, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    body_s  = ParagraphStyle("B",  parent=styles["Normal"], fontSize=9,  textColor=DARK, leading=14)
    disc_s  = ParagraphStyle("D",  parent=styles["Normal"], fontSize=7.5,textColor=GREY, alignment=TA_CENTER)
    bold_s  = ParagraphStyle("BL", parent=styles["Normal"], fontSize=9,  fontName="Helvetica-Bold")

    is_pd   = "Parkinson" in result.get("prediction", "")
    p_color = RED if is_pd else GREEN
    story   = []

    # ── Header ──
    story += [
        Paragraph("PD Detect — Voice Analysis Report", title_s),
        Paragraph("AI-Powered Parkinson's Disease Screening via Voice Biomarkers", sub_s),
        HRFlowable(width="100%", thickness=2, color=BLUE),
        Spacer(1, 5*mm),
    ]

    # ── Meta table ──
    meta = [
        ["Analysis ID", result.get("id", "—"),  "Timestamp", result.get("timestamp", "—")],
        ["Model",       result.get("model_used","ensemble"), "Risk Level", result.get("risk_level","—")],
    ]
    mt = Table(meta, colWidths=[35*mm, 60*mm, 35*mm, 40*mm])
    mt.setStyle(TableStyle([
        ("FONTNAME",   (0,0),(-1,-1), "Helvetica"),
        ("FONTNAME",   (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",   (2,0),(2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 9),
        ("TEXTCOLOR",  (0,0),(-1,-1), DARK),
        ("BACKGROUND", (0,0),(0,-1),  LIGHT),
        ("BACKGROUND", (2,0),(2,-1),  LIGHT),
        ("GRID",       (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
        ("PADDING",    (0,0),(-1,-1), 5),
    ]))
    story += [mt, Spacer(1, 5*mm)]

    # ── Result ──
    story.append(Paragraph("Diagnosis Result", h2_s))
    res_data = [
        ["Prediction", result.get("prediction","—")],
        ["Confidence", f"{result.get('confidence',0):.1f}%"],
        ["Risk Level", result.get("risk_level","—")],
    ]
    rt = Table(res_data, colWidths=[55*mm, 115*mm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(0,-1),  LIGHT),
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (1,0),(1,0),   "Helvetica-Bold"),
        ("FONTSIZE",    (1,0),(1,0),   13),
        ("TEXTCOLOR",   (1,0),(1,0),   p_color),
        ("GRID",        (0,0),(-1,-1), 0.4, colors.HexColor("#bfdbfe")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, LIGHT]),
        ("PADDING",     (0,0),(-1,-1), 6),
    ]))
    story += [rt, Spacer(1, 5*mm)]

    # ── Features ──
    story.append(Paragraph("Key Acoustic Features", h2_s))
    f = result.get("features", {})
    feat_rows = [["Feature", "Value", "Normal Range", "Status"]]
    feat_cfg  = [
        ("Jitter (Local)",         f.get("jitter_local"),       "0.001 – 0.010"),
        ("Shimmer (Local)",        f.get("shimmer_local"),      "0.010 – 0.060"),
        ("HNR (dB)",               f.get("hnr"),                "15 – 35"),
        ("Pitch Mean (Hz)",        f.get("pitch_mean"),         "75 – 300"),
        ("Pitch Std (Hz)",         f.get("pitch_std"),          "0.5 – 10"),
        ("ZCR Mean",               f.get("zcr_mean"),           "0.01 – 0.15"),
        ("RPDE",                   f.get("rpde"),               "0.40 – 0.65"),
        ("DFA",                    f.get("dfa"),                "0.50 – 0.80"),
        ("Spectral Centroid (Hz)", f.get("spectral_centroid_mean"), "500 – 4000"),
        ("RMS Energy",             f.get("rms_mean"),           "0.005 – 0.40"),
        ("Jitter (Abs)",           f.get("jitter_absolute"),    "< 0.00005"),
        ("Shimmer (dB)",           f.get("shimmer_db"),         "< 0.5"),
    ]
    for name, val, rng in feat_cfg:
        v_str  = f"{val:.5f}" if val is not None else "—"
        status = "—"
        if val is not None:
            parts = rng.replace("< ","0 – ").split(" – ")
            if len(parts) == 2:
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    status = "Normal" if lo <= val <= hi else "Abnormal"
                except:
                    pass
        feat_rows.append([name, v_str, rng, status])

    ft = Table(feat_rows, colWidths=[65*mm, 35*mm, 45*mm, 25*mm])
    style_cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bfdbfe")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT]),
        ("PADDING",      (0,0), (-1,-1), 4.5),
    ]
    for row_idx, (_, val, rng) in enumerate(feat_cfg, start=1):
        if val is not None:
            parts = rng.replace("< ","0 – ").split(" – ")
            status = "—"
            if len(parts) == 2:
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    status = "Normal" if lo <= val <= hi else "Abnormal"
                except:
                    pass
            c = GREEN if status == "Normal" else (RED if status == "Abnormal" else GREY)
            style_cmds.append(("TEXTCOLOR", (3, row_idx), (3, row_idx), c))
            style_cmds.append(("FONTNAME",  (3, row_idx), (3, row_idx), "Helvetica-Bold"))
    ft.setStyle(TableStyle(style_cmds))
    story += [ft, Spacer(1, 5*mm)]

    # ── Recommendations ──
    story.append(Paragraph("Recommendations", h2_s))
    for rec in result.get("recommendations", []):
        story.append(Paragraph(f"• {rec}", body_s))
    story.append(Spacer(1, 8*mm))

    # ── Disclaimer ──
    story += [
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8")),
        Spacer(1, 3*mm),
        Paragraph(
            "DISCLAIMER: This report is generated by an AI-based screening tool for research and educational purposes only. "
            "It does NOT constitute a medical diagnosis and has not been approved by any regulatory authority. "
            "Please consult a qualified neurologist for any medical concerns. "
            "Dataset: Little et al. (2007), UCI ML Repository.",
            disc_s,
        ),
    ]

    doc.build(story)
    return buf.getvalue()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "models_loaded": models_loaded,
        "history_count": len(analysis_history),
        "timestamp":     datetime.utcnow().isoformat(),
        "version":       "1.0.0",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if len(audio_bytes) < 200:
        raise HTTPException(status_code=400, detail="Audio file is empty or too small.")

    # Feature extraction
    try:
        features = extract_features(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Feature extraction failed: {str(e)}. Ensure audio contains clear speech.",
        )

    # Prediction
    if models_loaded and scaler is not None and ensemble is not None:
        try:
            uci_vec    = map_to_uci_vector(features).reshape(1, -1)
            uci_scaled = scaler.transform(uci_vec)
            proba      = float(ensemble.predict_proba(uci_scaled)[0][1])
            pred       = "Parkinson's Detected" if proba >= 0.5 else "Healthy"
            conf       = round((proba if proba >= 0.5 else 1.0 - proba) * 100, 1)
            model_used = "Ensemble (RF + XGBoost + SVM)"
        except Exception as e:
            logger.error(f"Model inference error: {e}. Falling back to heuristic.")
            p = heuristic_predict(features)
            pred, proba, conf, model_used = p["prediction"], p["probability"], p["confidence"], p["model_used"]
    else:
        p = heuristic_predict(features)
        pred, proba, conf, model_used = p["prediction"], p["probability"], p["confidence"], p["model_used"]

    recs = get_recommendations(pred, conf, features)
    fimp = build_feature_importance(features)

    result = {
        "id":                str(uuid.uuid4())[:8].upper(),
        "timestamp":         datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "prediction":        pred,
        "confidence":        conf,
        "probability":       round(proba, 4),
        "risk_level":        risk_level(proba),
        "features":          {k: round(float(v), 6) for k, v in features.items()},
        "feature_importance":fimp,
        "recommendations":   recs,
        "model_used":        model_used,
    }

    analysis_history.append(result)
    if len(analysis_history) > MAX_HISTORY:
        analysis_history.pop(0)

    return result


@app.get("/history")
async def get_history():
    return {
        "history": list(reversed(analysis_history)),
        "count":   len(analysis_history),
    }


@app.delete("/history")
async def clear_all_history():
    analysis_history.clear()
    return {"message": "History cleared.", "count": 0}


@app.delete("/history/{entry_id}")
async def delete_entry(entry_id: str):
    global analysis_history
    before = len(analysis_history)
    analysis_history = [h for h in analysis_history if h.get("id") != entry_id]
    if len(analysis_history) == before:
        raise HTTPException(status_code=404, detail="Entry not found.")
    return {"message": f"Entry {entry_id} deleted."}


@app.get("/stats")
async def get_stats():
    if not analysis_history:
        return {
            "total":           0,
            "pd_count":        0,
            "healthy_count":   0,
            "avg_confidence":  0,
            "feature_averages":{},
            "risk_distribution":{"High": 0, "Medium": 0, "Low": 0},
        }

    pd_list = [h for h in analysis_history if h["prediction"].startswith("Parkinson")]
    he_list = [h for h in analysis_history if not h["prediction"].startswith("Parkinson")]

    # Feature averages
    feat_keys = ["jitter_local","shimmer_local","hnr","pitch_mean","pitch_std",
                 "zcr_mean","rpde","dfa","rms_mean","spectral_centroid_mean"]
    feat_avgs = {}
    for k in feat_keys:
        vals = [h["features"].get(k) for h in analysis_history if h["features"].get(k) is not None]
        feat_avgs[k] = round(float(np.mean(vals)), 6) if vals else 0.0

    # Risk distribution
    risk_dist = {"High": 0, "Medium": 0, "Low": 0}
    for h in analysis_history:
        r = h.get("risk_level", "Low")
        risk_dist[r] = risk_dist.get(r, 0) + 1

    return {
        "total":             len(analysis_history),
        "pd_count":          len(pd_list),
        "healthy_count":     len(he_list),
        "avg_confidence":    round(float(np.mean([h["confidence"] for h in analysis_history])), 1),
        "feature_averages":  feat_avgs,
        "risk_distribution": risk_dist,
        "models_loaded":     models_loaded,
    }


@app.post("/report")
async def generate_report(request: Request):
    """Accept raw JSON body for PDF generation."""
    try:
        result = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    try:
        pdf_bytes = build_pdf(result)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"pd_report_{ts}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
