# Exoplanets_NASA_2025 — README

<img width="1230" height="291" alt="Screenshot 2025-10-08 at 2 35 57 PM" src="https://github.com/user-attachments/assets/2edfac13-89a0-4b87-b4c6-4328a622472f" />


This repository contains a trimmed and portable copy of the NASA Space Apps Exoplanet project. It is split into two main deliverables:

- `Model Web/` — a compact Streamlit-based demo (Comprehensive Exoplanet Hunter AI). This is the primary demo you can run locally. It includes a small fallback path so the UI runs even without the original large trained models.
- `Web app/` — a React + Vite frontend with a Flask backend that serves analysis images and a small dataset. This is an interactive visualization and exploration UI.

This README explains the purpose of each folder, how to run the Streamlit demo, how to run the React+Flask web app, and how to restore large archived model files if you need them locally.

---

## Quick navigation
- Model Web (Streamlit demo): `Model Web/app/Comprehensive_NASA_Space_Apps_App.py`
- Web app (React + Flask): `Web app/` (frontend + `Web app/server/` backend)
- Local archive (NOT pushed to GitHub): `archive_pruned/` — contains large trained `*.joblib` model files and raw dataset CSVs. These were intentionally kept out of the remote repo to reduce size.

---

## Model Web (Streamlit demo)

Overview
- A trimmed Streamlit demo that recreates the core UI from the NASA Space Apps project. It accepts transit and stellar features, runs a classification model, and shows probabilities, metrics, and sample dataset rows.
- The demo is intentionally lightweight and includes a fallback mechanism: when the expected large model is absent, the app constructs a tiny logistic-regression fallback so the UI remains interactive.

Contents (important files)
- `Model Web/app/Comprehensive_NASA_Space_Apps_App.py` — the main Streamlit application.
- `Model Web/models/` — keep small JSON summaries (e.g., `*_summary.json`) here. Large `*.joblib` model binaries are stored locally in `archive_pruned/` and are NOT pushed to GitHub.
- `Model Web/data/processed/` — small processed sample files (parquet/csv/json) used to populate UI tables and sample metrics.
- `Model Web/simple_train.py`, `Model Web/train_exoplanet_model.py`, `Model Web/train_enhanced_exoplanet_model.py` — minimal training scripts kept to reproduce or train smaller models.
- `Model Web/trim_code_for_github.sh`, `prune_for_github.sh` — utility scripts used to produce the pruned repository and move big artifacts to `archive_pruned/`.

What the demo does
- Presents an input form for Basic / Advanced / Expert features.
- Loads a saved model from `Model Web/models/` if present. Expected model format (joblib) is a dict-like object with keys: `model`, `scaler`, `feature_columns`, and optional `threshold`/`training_results`.
- If no model file is found, builds a tiny fallback LogisticRegression model with sensible defaults so the UI remains functional.
- Shows a small sample dataset and simple visualizations (Plotly) for probabilities and feature importances.

Run the demo (minimal)
1. Create and activate a Python virtual environment (or use your existing one):

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (fast minimal set):

```bash
pip install -r "Model Web/requirements.txt"
# or a minimal set:
pip install numpy pandas scikit-learn joblib plotly streamlit xgboost
```

3. Run the Streamlit app from the `Model Web` folder:

```bash
cd "Model Web"
streamlit run app/Comprehensive_NASA_Space_Apps_App.py --server.port 5782
```

4. Open your browser at: `http://localhost:5782`

Notes
- If you want to use a real trained model, copy the model `*.joblib` (with the expected dict structure) into `Model Web/models/` before starting the app. Example expected structure:

```python
{
  'model': sklearn_estimator,
  'scaler': sklearn_scaler,
  'feature_columns': ['period_days', 'duration_hours', ...],
  'threshold': 0.5
}
```

- If you don't have the large models available, the fallback lets reviewers open the UI and test inputs.

Troubleshooting
- If Streamlit complains about ports in use, pick a different `--server.port` value or free the port shown in the log.
- If model load fails due to pickling/version issues, verify scikit-learn/xgboost versions used to create the model.

---

## Web app (React + Vite frontend + Flask backend)

Overview
- `Web app/` is a separate interactive frontend that presents the exoplanet catalog, per-planet detail pages, and static analysis images (PNG figures) produced by analysis scripts.
- The UI uses Vite + React for a snappy local development experience; the Python Flask backend serves the dataset and the `server/output/` images.

Contents (important files)
- `Web app/src/` — React frontend source.
- `Web app/public/` — static public assets used by the frontend.
- `Web app/server/` — Flask backend, analysis scripts, and pre-generated images in `server/output/`.
- `.env` — local environment config for Vite/Front-end keys (e.g., `VITE_GEMINI_KEY`). Do NOT commit secrets.

Run the Web app locally (dev flow)
Requirements:
- Node.js 16+ and npm or pnpm
- Python 3.11+ (or use the provided `server/nasa25` virtualenv)

1) Start the frontend (Vite)

```bash
cd "Web app"
# install dependencies
npm install
# ensure .env is configured (VITE_GEMINI_KEY if you use any API-dependent features)
npm run dev
```

Vite will print the local frontend URL (commonly `http://localhost:5173`).

2) Start the backend (Flask)

Open a second terminal:

```bash
cd "Web app/server"
# Option A: use your system Python
python3 main.py

# Option B: activate the provided virtualenv (recommended)
source nasa25/bin/activate
python main.py
```

Default backend port: 5500. If you must change the port set `PORT` env var before running.

Verify endpoints
- Images: `http://localhost:5500/output/fig1_linear_regression.png`
- API endpoints: check `server/routes/*.py` for endpoints the frontend consumes.

Troubleshooting and notes
- If the frontend shows missing images, ensure Flask is running and `server/output/` files exist.
- macOS: avoid port 5000 due to system services — the backend defaults to 5500.
- If you regenerate figures, run the analysis scripts in `server/` to repopulate `server/output/`.

---

## Restoring large models (local only)

Large model files and full raw datasets were intentionally archived locally to keep the GitHub repository small.
They are located in your local `archive_pruned/` folder. To restore a model for local use:

```bash
# copy the comprehensive model back into Model Web/models/
cp archive_pruned/comprehensive_best_model.joblib "Model Web/models/"

# then run the Streamlit demo again
cd "Model Web"
streamlit run app/Comprehensive_NASA_Space_Apps_App.py --server.port 5782
```

If you want large models on the remote, use Git LFS. I can help set that up.

---

## GitHub, LFS, and publishing notes
- The repository kept large model files out of the remote history to avoid hitting GitHub limits. If you need those files persisted remotely, enable Git Large File Storage (LFS):

```bash
git lfs install
git lfs track "archive_pruned/*.joblib"
git add .gitattributes
git add archive_pruned/*.joblib
git commit -m "Track large models with git-lfs"
git push origin main
```

Note: Git LFS usage may incur storage or bandwidth limits depending on your account.

---

## Development & contributions
- To re-run data processing or retrain models, explore the scripts in `Model Web/src/` (data processing and feature engineering) and `Model Web/src/models/` (trainers). Many helper scripts were archived in `archive_pruned/code_archive/` — move any you need back into the project to re-run training pipelines.
- If you want, I can produce a small demo model and update the Streamlit app to auto-use it when no large model is available.

## License
- Add your preferred LICENSE file before publishing this repo to make licensing explicit (e.g., MIT).

---


