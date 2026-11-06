# Speech Emotion Recognition from Acoustic Features using a Multi-Layer Perceptron Classifier

An affective computing prototype and baseline ML system classifying 4 discrete emotional states (`calm`, `happy`, `fearful`, `disgust`) from speech audio using 180 acoustic features (`MFCC 40` + `Chroma STFT 12` + `Mel Spectrogram 128`) and a lightweight Multi-Layer Perceptron (MLP) Classifier. 

Mapped to **UN Sustainable Development Goal 3 (Good Health and Well-Being)**, Target 3.4 (Mental Health & Well-being).

---

## 1. Directory Structure

```text
ml project/
├── backend/
│   ├── __init__.py
│   ├── config.py             # Feature dimensions, paths, 4-class mappings
│   ├── extractor.py          # Librosa 180-dim feature extractor (MFCC + Chroma + Mel)
│   ├── train.py              # ML pipeline: extraction, MLP fit, metrics, real EDA JSON
│   └── server.py             # FastAPI REST API & static web server
├── frontend/
│   ├── index.html            # 5-page SPA (Overview, Dataset, EDA, Model & Results, Predict)
│   ├── styles.css            # Light-theme design system, responsive layout, heatmap styles
│   └── app.js                # State management, Chart.js visualisations, audio prediction
├── data/
│   └── Audio_Speech_Actors_01-24/ # Expected location of RAVDESS audio speech dataset
├── models/
│   ├── mlp_model.joblib      # Serialized trained MLP model
│   └── metrics_eda.json      # Computed real metrics, confusion matrix, EDA summaries
├── requirements.txt          # Python dependencies
└── README.md                 # Setup and run instructions
```

---

## 2. Dataset Setup

This baseline strictly uses the audio-only speech portion of the **RAVDESS** dataset (**1,440 files across 24 actors**):
1. Place or extract the `Audio_Speech_Actors_01-24` directory into:
   ```text
   ./data/Audio_Speech_Actors_01-24/
   ```
2. Verify that `Actor_01`, `Actor_02`, ..., `Actor_24` folders are present under `./data/Audio_Speech_Actors_01-24/`.
3. The training pipeline will automatically scan all files and filter strictly for the 4 target emotions:
   - `02` → `calm` (192 files)
   - `03` → `happy` (192 files)
   - `06` → `fearful` (192 files)
   - `07` → `disgust` (192 files)
   Total: **768 files** (balanced 192 per class).

---

## 3. Installation & Environment Setup

Ensure Python 3.10+ is installed.

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

---

## 4. Running the Machine Learning Pipeline

Execute the pipeline to extract features, train the MLP model, evaluate metrics, and export EDA artifacts:

```bash
python backend/train.py
```

### What happens during training:
- Loads the 768 audio files and extracts 180 unscaled features per file (`MFCC 40` + `Chroma 12` + `Mel 128`).
- Evaluates descriptive statistics (mean, median, std, min, max), class distributions, and correlation matrices.
- Splits data 75:25 using `train_test_split(..., test_size=0.25, random_state=9)`.
- Fits `MLPClassifier(hidden_layer_sizes=(300,), learning_rate='adaptive', random_state=9)`.
- Calculates Accuracy, Macro Precision, Macro Recall, Macro F1, and OvR AUC-ROC on both Train and Test sets.
- Generates the 4x4 confusion matrix, per-class metrics, `fearful` recall highlight, and automated interpretation.
- Saves `models/mlp_model.joblib` and `models/metrics_eda.json`.

---

## 5. Launching the Web Application

Start the FastAPI application:

```bash
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```text
http://localhost:8000
```

---

## 6. Frontend Features (The 5 Spec Pages)

1. **Overview**: Project title, problem statement, research question, hypothesis (falsifiability criterion), and SDG 3 / Target 3.4 mapping.
2. **Dataset**: RAVDESS description, 4-class subset breakdown, interactive data dictionary table, and documented preprocessing steps.
3. **EDA**: Real descriptive statistics table, class balance bar chart (Chart.js), interactive MFCC distribution curves, correlation heatmap, and synthesized observations.
4. **Model & Results**: Baseline architecture specifications, train-vs-test comparison table, confusion matrix heatmap, per-class performance with dedicated `fearful` recall callout, and auto-generated analytical interpretation.
5. **Predict**: Drag-and-drop `.wav` audio upload, waveform player preview, feature extraction, and real-time MLP classification with class probability distribution bars.
