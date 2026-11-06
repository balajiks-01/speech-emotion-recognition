from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "Audio_Speech_Actors_01-24"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "mlp_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics_eda.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# Target 4-Class subset
TARGET_EMOTIONS = {
    "02": "calm",
    "03": "happy",
    "06": "fearful",
    "07": "disgust"
}

EMOTION_LABELS = sorted(list(TARGET_EMOTIONS.values()))  # ['calm', 'disgust', 'fearful', 'happy']
EMOTION_TO_CODE = {v: k for k, v in TARGET_EMOTIONS.items()}

# Feature extraction dimensions
N_MFCC = 40
N_CHROMA = 12
N_MELS = 128
TOTAL_FEATURES = N_MFCC + N_CHROMA + N_MELS  # 180

# Training configuration
TEST_SIZE = 0.25
RANDOM_STATE = 9
MLP_HIDDEN_LAYER_SIZES = (300,)
MLP_LEARNING_RATE = "adaptive"
