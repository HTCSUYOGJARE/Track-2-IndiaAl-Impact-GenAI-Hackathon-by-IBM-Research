# SmartWatt — Short-Term Load Forecasting (IBM Research Hackathon, Track-2)

This repository contains my end-to-end pipeline and final solution for the **IBM Research – Short-Term Load Forecasting** competition on Kaggle (Track-2 of IndiaAI Impact Gen-AI Hackathon).

- Competition page: https://www.kaggle.com/competitions/short-term-load-forecasting/overview  
- Task: Predict the next **24 hours** of electricity consumption for residential/commercial buildings given the previous **168 hours**.  
- Metric: **Mean Squared Error (MSE)**.  
- Result: **4th place** overall.

---

## 1) Project Overview

Buildings account for ~⅓ of global energy use and emissions. Accurate short-term load forecasting (STLF) supports demand response, efficient chiller scheduling, and renewable integration. This repo implements **SmartWatt**, a lightweight, reproducible STLF pipeline centered on **Granite Time-Series Foundation Models (TSFMs)**, with careful preprocessing and (optional) seed ensembling for robust generalization.

---

## 2) Dataset

Use the Kaggle dataset directly (do not commit data here):

- Train/Test: hourly meter readings organized in 192-hour windows (`role ∈ {input, target}`), plus `window_id`, `building_id`, `timestamp`, and `meter_reading`.
- Metadata: static attributes per `building_id` (region, area, occupants, appliances, etc.).

Download instructions are on the Kaggle page above.

---

## 3) Approach & Methodology

### 3.1 Data Preparation (see `Data_preparation.ipynb`)
- **Merge**: Join `train.csv`/`test.csv` with `metadata.csv` on `building_id` (deduplicated).
- **Imputation**:
  - Numeric (e.g., `area_in_sqft`) via **HistGradientBoostingRegressor** (fallback to median).
  - Binary/categorical (e.g., `inverter`) via **RandomForestClassifier** when present.
- **Time features**: `hour`, `dayofweek`, `month`, `is_weekend`.
- **Temporal features** (per building):
  - **Lags**: e.g., `meter_lag1` (and similar, as available in the notebook).
  - **Rolling stats**: e.g., `meter_roll_mean_24h`, `meter_roll_std_24h`, `meter_roll_mean_7d`.
- **Static ratios**:
  - `area_per_room`, `people_per_room`, `people_per_sqft`,
  - `lights_per_room`, `fans_per_room`, `ac_per_room`, etc.
  - `appliance_density` over `area_in_sqft`.
- **Outputs**: writes **`final_train.csv`** and **`final_test.csv`** (downstream model inputs).

> Notes:
> - `building_id` is kept as string and trimmed.
> - Strict dtype control to avoid `object`→torch issues.

### 3.2 Modeling & Training (see `Final_submission.ipynb`)
- **Backbone**: `ibm-granite/granite-timeseries-ttm-r2` via TSFM public toolkit.
- **Horizon & Context**: `prediction_length=24`, `context_length=168`.
- **Preprocessor**: TSFM `TimeSeriesPreprocessor` (per-series normalization handled internally).
- **Fine-tuning schedule**:
  - The notebook is set up for a **lightweight finetune** using HuggingFace `Trainer` with early stopping.
  - Default seed is set (e.g., `SEED=79`). You can duplicate a cell with different seeds to produce multiple submissions.
- **Submission**: predictions are written to `submission_seed_{SEED}_r2.csv`.
- **(Optional) Ensembling**:
  - There is commented code for **convex MSE-only blending** across per-seed submissions (simplex weights learned on validation; applied to test). Enable it if you generate multiple seed submissions.

---

## 4) Results

- Leaderboard: **4th place**.  
- Model family: TTM-r2 with a compact trainable head/decoder (encoder remains frozen for speed and stability).  
- Observations:
  - Clean preprocessing and leak-safe handling of IDs strongly stabilized validation.
  - Simple MSE-only ensembling across seeds improved robustness on the hidden leaderboard.

*(Fill in your public/private MSE if you’d like.)*

---

## 5) Repository Structure

```
.
├── Data_preparation.ipynb      # Build final_train.csv / final_test.csv from Kaggle CSVs + metadata
├── Final_submission.ipynb      # Train/eval with Granite TTM-r2 and write submission CSV(s)
├── requirements.txt            # Exact Python dependencies
├── README.md                   # This file
└── (data is not committed; download from Kaggle)
```

Generated artifacts (not tracked):
```
final_train.csv
final_test.csv
submission_seed_{SEED}_r2.csv
```

---

## 6) How to Reproduce

### 6.1 Environment
```bash
python -m venv .venv
source .venv/bin/activate           # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 6.2 Data
1. Download the Kaggle dataset from the competition page.
2. Keep `train.csv`, `test.csv`, `metadata.csv`, `sample_submission.csv` together locally.
3. Open and run **`Data_preparation.ipynb`** end-to-end (adjust paths in the first cell if needed).  
   It will output:
   - `final_train.csv`
   - `final_test.csv`

### 6.3 Train & Predict
1. Open **`Final_submission.ipynb`**.
2. Ensure the constants at the top point to:
   - `TRAIN_FILE_PATH = "final_train.csv"`
   - `TEST_FILE_PATH  = "final_test.csv"`
3. Run all cells to produce a submission CSV:
   - `submission_seed_{SEED}_r2.csv`  
4. (Optional) Re-run with different `SEED`s and enable the blending cell to create an averaged submission.

---

## 7) Design Choices & Ablations

- **Foundation model**: Granite **TinyTimeMixer (TTM-r2)**—compact head/decoder finetuning with an otherwise frozen encoder to keep training light and stable for hackathon timelines.
- **Windowing**: Fixed `168→24` aligns with task definition and simplifies batching/masking.
- **Normalization**: Rely on TSFM’s internal per-series scaling; avoid ad-hoc external scalers that can cause leakage.
- **Leakage guardrails**: Fit categorical encodings/statistics strictly on train partitions; keep `building_id` string-typed.
- **Ensembling**: MSE-only convex blend of seed predictions; avoids overfitting from exotic per-horizon calibration.

---

## 8) Tips for Extending

- **More seeds**: Generate 3–5 per family and blend.
- **Targeted micro-unfreeze**: Briefly unfreeze the last encoder block with tiny LR; keep early stopping tight.
- **Residual stacking**: Fit a tiny regressor on validation residuals using simple exogenous features (hour/region).
- **Model diversity**: Add PatchTSMixer (TSFM public) as a third family for a 3-way blend.

---

## 9) Competition Summary (Track-2)

**Problem**  
24-hour ahead point forecasting of building electricity demand from 168 hours of context, across diverse regions/building types with realistic noise and operational variability. Daily rolling windows simulate real deployment.

**Objective**  
Build accurate, generalizable STLF models; leverage/fine-tune Time Series Foundation Models; explore robustness to cycles, seasonality, and anomalies.

**Evaluation**  
Submissions are scored by **MSE** between predicted and observed meter readings. Submission format:

```
row_id,meter_reading
169,0
170,0
...
```

**Acknowledgement**  
Dataset courtesy of the **Prayas, Energy Group**, provided via Kaggle competition.

---

## 10) How This Repo Helps Reviewers

- **Reproducible**: `Data_preparation.ipynb` → `final_train.csv`/`final_test.csv` → `Final_submission.ipynb` → submission CSV.
- **Auditable**: Clear feature engineering (lags/rolls/static ratios) and transparent TSFM training loop.
- **Lightweight**: Fast to iterate; suitable for hackathon constraints yet competitive on leaderboard.

---

## 11) License

Choose an OSI license (e.g., MIT or Apache-2.0) and place it in `LICENSE`. Until then, “All rights reserved.”

---

## 12) Acknowledgements

- IBM Research and Kaggle organizers for the competition and baseline resources.
- Prayas, Energy Group for the dataset.

---

### Notes for Maintainers

- If you move data to a `data/` folder, adjust the paths in the first cells of both notebooks.
- The blending cell in `Final_submission.ipynb` is intentionally commented. Uncomment it only after you have at least two submission files from different seeds.
