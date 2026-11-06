import json
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import (
    MODEL_FILE,
    METRICS_FILE,
    FRONTEND_DIR,
    EMOTION_LABELS,
    TARGET_EMOTIONS
)
from backend.extractor import extract_features_from_bytes

app = FastAPI(
    title="Speech Emotion Recognition MLP API",
    description="Acoustic Feature Extraction & MLP Classifier Baseline for SER",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_metrics_cache():
    """Load cached metrics and EDA generated from the real training run."""
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_trained_model():
    """Load trained MLP model."""
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)

@app.get("/api/status")
def get_system_status():
    """Check if model and real metrics are available."""
    model_loaded = MODEL_FILE.exists()
    metrics_loaded = METRICS_FILE.exists()
    return {
        "status": "ready" if (model_loaded and metrics_loaded) else "needs_training",
        "model_file_exists": model_loaded,
        "metrics_file_exists": metrics_loaded
    }

@app.get("/api/overview")
def get_overview():
    return {
        "title": "Speech Emotion Recognition from Acoustic Features using a Multi-Layer Perceptron Classifier",
        "domain": "Affective Computing / Applied Machine Learning",
        "sdg_mapping": {
            "goal": "SDG 3: Good Health and Well-Being",
            "target": "Target 3.4 (Mental Health & Well-being)",
            "description": "In telehealth consultations and remote crisis helplines, subtle vocal distress cues often go undetected until a critical escalation occurs. Automated SER provides an objective, real-time alert system to prioritise urgent calls and assist counsellors under heavy caseloads."
        },
        "problem_statement": "Human speech carries acoustic indicators of affective state (timbre, harmonics, pitch variance). Conventional triage relies on subjective listener perception or rigid rules. This project investigates whether raw acoustic features can directly infer discrete emotional categories to power automated distress detection in healthcare environments.",
        "research_question": "Can acoustic features extracted from a raw speech signal — specifically Mel-Frequency Cepstral Coefficients (MFCC), Chroma, and Mel-Spectrogram energy — be used to reliably classify a speaker's discrete emotional state (calm, happy, fearful, disgust) using a lightweight Multi-Layer Perceptron classifier, and how does the resulting baseline accuracy compare against the more complex deep-learning approaches reported in the literature?",
        "hypothesis": "We hypothesise that MFCC, Chroma, and Mel-spectrogram features contain sufficient discriminative information for a simple MLP classifier to distinguish between the four target emotion classes (calm, happy, fearful, disgust) in the RAVDESS dataset at an accuracy meaningfully above the random-guess baseline of 25%. We further hypothesise that acoustically similar emotion pairs (for example, fearful vs. disgust, both characterised by tense, lower-energy vocalisation) will show a higher rate of mutual misclassification than acoustically distinct pairs (for example, happy vs. fearful, which differ sharply in pitch variance and energy). This hypothesis is falsifiable via the confusion matrix."
    }

@app.get("/api/dataset")
def get_dataset_info():
    return {
        "name": "Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)",
        "license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        "actors": 24,
        "actor_distribution": "12 male actors (odd IDs), 12 female actors (even IDs)",
        "total_speech_files": 1440,
        "working_subset": {
            "selected_emotions": ["calm", "happy", "fearful", "disgust"],
            "total_samples": 768,
            "samples_per_class": 192,
            "filename_format": "[modality]-[vocal_channel]-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav"
        },
        "data_dictionary": [
            {
                "feature_name": "MFCC (mean, 40 coefficients)",
                "data_type": "float[40]",
                "description": "Mel-Frequency Cepstral Coefficients averaged over time frames; captures the short-term spectral envelope (timbre) of the speech signal."
            },
            {
                "feature_name": "Chroma STFT (mean, 12 bins)",
                "data_type": "float[12]",
                "description": "Pitch-class energy distribution computed from Short-Time Fourier Transform; captures harmonic/pitch content."
            },
            {
                "feature_name": "Mel Spectrogram (mean, 128 bins)",
                "data_type": "float[128]",
                "description": "Frequency-domain energy representation on the mel scale, approximating human auditory perception."
            },
            {
                "feature_name": "Sample Rate",
                "data_type": "integer",
                "description": "Native sampling rate of each .wav file (Hz); used internally during feature extraction."
            },
            {
                "feature_name": "Emotion Label (target)",
                "data_type": "categorical",
                "description": "Ground-truth emotion class parsed from the filename encoding: 02=calm, 03=happy, 06=fearful, 07=disgust."
            },
            {
                "feature_name": "Actor ID",
                "data_type": "categorical",
                "description": "Identifier (01-24) of the actor who recorded the sample; used to check speaker balance."
            }
        ],
        "preprocessing_steps": [
            {
                "step": "Missing Value Handling",
                "detail": "All 768 audio files in the 4-class subset verified to load without I/O errors. No missing or corrupted samples."
            },
            {
                "step": "Encoding of Categorical Variables",
                "detail": "Emotion labels parsed directly from the 3rd field of the standardized 7-part filename. Scikit-learn MLPClassifier performs internal target mapping."
            },
            {
                "step": "Feature Scaling / Normalisation",
                "detail": "Not applied in this baseline — MFCC, Chroma, and Mel features were used exactly as extracted (time-averaged) by librosa. Documented as a known baseline limitation."
            },
            {
                "step": "Outlier Treatment",
                "detail": "No explicit outlier removal was performed on extracted feature vectors for this baseline."
            },
            {
                "step": "Train/Test Split",
                "detail": "75:25 train-test split applied using scikit-learn train_test_split with fixed random_state=9 (576 training, 192 testing)."
            }
        ]
    }

@app.get("/api/eda")
def get_eda():
    data = load_metrics_cache()
    if not data or "eda" not in data:
        raise HTTPException(
            status_code=404,
            detail="EDA metrics not generated yet. Please run the training pipeline first: python backend/train.py"
        )
    return data["eda"]

@app.get("/api/results")
def get_results():
    data = load_metrics_cache()
    if not data or "results" not in data:
        raise HTTPException(
            status_code=404,
            detail="Model results not generated yet. Please run the training pipeline first: python backend/train.py"
        )
    return data["results"]

@app.post("/api/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Accept an uploaded .wav file, extract the 180 acoustic features,
    and predict the emotion using the trained MLPClassifier.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav audio files are supported.")
    
    model = load_trained_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Please run the training pipeline first: python backend/train.py"
        )
    
    try:
        content = await file.read()
        features, summary = extract_features_from_bytes(content)
        
        # Reshape to (1, 180)
        X_sample = np.array(features, dtype=np.float32).reshape(1, -1)
        
        prediction = model.predict(X_sample)[0]
        probabilities = model.predict_proba(X_sample)[0]
        
        class_probs = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(model.classes_, probabilities)
        }
        
        confidence = float(np.max(probabilities))
        
        # Emotion descriptions for context
        descriptions = {
            "calm": "Speaker displays relaxed vocal tone, steady pitch, and low acoustic energy.",
            "happy": "Speaker exhibits elevated pitch variance, brighter timbre, and energetic modulation.",
            "fearful": "Speaker indicates vocal tension, tremor, and compressed harmonic distribution (distress marker).",
            "disgust": "Speaker displays lower-energy vocal resonance with restricted frequency modulation."
        }
        
        return {
            "filename": file.filename,
            "predicted_emotion": prediction,
            "confidence": round(confidence * 100, 2),
            "description": descriptions.get(prediction, ""),
            "probabilities": class_probs,
            "audio_summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction or prediction error: {str(e)}")

# Mount frontend directory for static UI serving
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
