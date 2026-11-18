# PathSnap — Team 11

> GPS-to-road map-matching: take noisy GPS points, generate candidate roads, score them with an ML classifier, rebuild the route with HMM/Viterbi, and serve it through a FastAPI backend and a Streamlit UI.

---

## 📂 Repository Structure

```
pathsnap-team11/
│
├── common/                  # Role A — Shared data loader, path config, projection helpers
├── data_processing/         # Role A — Raw-data inspection, preprocessing, cleaning
├── features/                # Role A — Feature engineering, candidate generation
├── visualisation/           # Role A — Folium maps, plotting utilities
│
├── models/                  # Roles B & C — Model training scripts (all 5+ models)
├── optimisation/            # Role B — Bayesian hyperparameter optimisation (Optuna)
├── results/                 # Role B — Experiment logs, comparison tables, plots
│
├── decoder/                 # Role C — HMM + Viterbi route reconstruction
├── artifacts/               # Role C — Final model weights, feature_list.json, HMM config
│
├── api/                     # Role D — FastAPI backend
├── ui/                      # Role D — Streamlit evaluator UI
├── report/                  # Role D — Technical report (PDF + assets)
│
├── data/                    # Data directory (git-ignored, see Setup below)
│   ├── train/               # 80 training trips
│   ├── student_test/        # 10 student-testing trips
│   └── evaluator_test/      # 10 evaluator-testing trips (DO NOT TOUCH)
│
├── notebooks/               # Exploratory notebooks
├── requirements.txt         # Pinned dependencies (Python 3.11)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/pathsnap-team11.git
cd pathsnap-team11
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Add data

Copy the dataset folders into `data/`:

```
data/
├── train/
│   ├── 00000001/
│   │   ├── 00000001.track
│   │   ├── 00000001.nodes
│   │   ├── 00000001.arcs
│   │   └── 00000001.route
│   └── ...
└── student_test/
    ├── 00000000/
    └── ...
```

### 3. Run the API (Laptop 1)

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Run the UI (Laptop 2)

```bash
cd ui
streamlit run app.py -- --api-url http://<LAPTOP1_IP>:8000
```

---

## 🏗️ Pipeline Overview

```
GPS Track (.track)
      │
      ▼
Data Preprocessing & Candidate Generation
      │
      ▼
Feature Engineering → Candidate Table
      │
      ▼
ML Classifier (predict P(correct) per candidate)
      │
      ▼
HMM + Viterbi Decoder (route reconstruction)
      │
      ▼
Predicted Route (sequence of arc IDs)
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| CV Macro F1 | Cross-validated macro F1 on training data |
| Top-1 Accuracy | Observation-level accuracy on student-test |
| Macro F1 | On student-testing dataset |
| Log Loss | Where applicable |
| Route Continuity | Connectivity of predicted route |
| Inference Time | End-to-end prediction time |

---

## 🏷️ Team Roles

| Role | Member | Responsibility |
|------|--------|---------------|
| A | Arti | Data, Features & Maps |
| B | Chiranth | Experiments & Bayesian Tuning (Models 1–3) |
| C | Akhilesh | Models 4–5 & HMM/Viterbi |
| D | Adarsh | API, UI, Deployment & Submission |

---

## 📝 Branch & Commit Convention

- Each role works on their own branch: `role-a`, `role-b`, `role-c`, `role-d`
- Merge into `main` via pull requests
- Commit only inside your own folders
- Never commit secrets, API keys, or passwords

---

## ⚠️ Important Rules

- **Never** use evaluator-testing data during development
- **Never** hardcode predictions or bypass the API
- Fit all preprocessing transformations on training data only
- Ground truth stays server-side — never exposed to the prediction model or UI
- All reported numbers must come from actual code execution

---

## 📄 License

Academic project — PathSnap Hackathon 2026.
