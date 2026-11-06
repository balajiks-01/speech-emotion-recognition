import io
import numpy as np
import librosa
from backend.config import N_MFCC, N_MELS

def extract_features_from_audio(y, sr):
    """
    Extract 180 acoustic features from audio signal:
    - 40 MFCCs (mean over time)
    - 12 Chroma STFT bins (mean over time)
    - 128 Mel-spectrogram bands (mean over time)
    Total = 180 features
    """
    # 1. MFCC (40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # 2. Chroma STFT (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # 3. Mel Spectrogram (128)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_mean = np.mean(mel.T, axis=0)

    # Combined 180-dim feature vector
    features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
    return features, mfcc_mean, chroma_mean, mel_mean

def extract_features(file_path):
    """
    Load an audio file from disk and extract 180 features.
    """
    y, sr = librosa.load(file_path, sr=None)
    features, _, _, _ = extract_features_from_audio(y, sr)
    return features

def extract_features_from_bytes(file_bytes):
    """
    Extract 180 features from in-memory audio bytes, along with audio summary stats.
    """
    audio_io = io.BytesIO(file_bytes)
    y, sr = librosa.load(audio_io, sr=None)
    duration = float(librosa.get_duration(y=y, sr=sr))
    features, mfcc_mean, chroma_mean, mel_mean = extract_features_from_audio(y, sr)
    
    summary = {
        "duration_seconds": round(duration, 2),
        "sample_rate_hz": int(sr),
        "n_samples": int(len(y)),
        "mean_mfcc_energy": round(float(np.mean(mfcc_mean)), 4),
        "mean_chroma_energy": round(float(np.mean(chroma_mean)), 4),
        "mean_mel_energy": round(float(np.mean(mel_mean)), 4)
    }
    return features, summary
