# Track-2-IndiaAl-Impact-GenAI-Hackathon
Hackathon files for reference
## Project Story — About the Project  SmartWatt

### Inspiration
In our department building, HVAC, lighting, and lab machines often stay on after hours. Cutting power by floor reduces waste but breaks experiments, overnight computations, and early classes. That small pain reflects a national need: **reduce building energy use without blunt shutdowns**. This motivated **SmartWatt**—an AI system that forecasts short-term load and triages anomalies that operators can trust.

### What it does
   1) SmartWatt delivers short-term (24-hour) building-load forecasts from 168 hours of context, with few-shot transfer  on Indian buildings.
   2) Uses pretrained IBM Granite Time Series – TinyTimeMixer (TTM) backbones (both r1 and r2 checkpoints), fine-tuned on your windowed data.
  3) Produces robust submissions via seed ensembling and cross-family blending (R1+R2) for better generalization to the hidden leaderboard.
   4) Keeps the model simple & stable for hackathon runtime: encoder frozen, decoder+head trained, RPT disabled, batch=32 (GA=2).

### Approach
1) Data → Features → Windows
--Construct 192-step windows per building/region with roles (input:168 / target:24).
--Add calendar features: hour, dayofweek, month, hour_sin, hour_cos.
--One-hot region using train-only categories to avoid leakage; median-impute numeric controls.
--Enforce float32 to prevent object→torch errors.

2) Preprocessing
--Use TimeSeriesPreprocessor to format series; leave model scaling on (TTM’s built-in scaler handles per-series normalization internally).
--Pad/crop to ckpt CL=512 in a collator; mask pads; (RPT kept off after ablation).

3) Modeling
--Load TinyTimeMixerForPrediction from ibm-granite/granite-timeseries-ttm-r1 and …-r2.
--Align channel counts to dataset; prune head from ckpt FL→24 without touching backbone geometry.

 ---2-phase fine-tuning:
        --Phase-1 (head-only): lr_head=1e-3, ~8 epochs, ES on val MSE.
        --Phase-2 (decoder+head): lr_head=8e-4, lr_dec=2e-4 (also tried 6e-4/1.5e-4), ~20–30 epochs, warmup 7–10%,    head dropout=0.2.
        --Encoder frozen (micro-unfreeze considered but not required for current gains).

4) Validation & Ensembling
-- Save per-seed validation preds and test submissions.
-- Convex MSE blending across seeds within each family (weights on simplex, auto-downweights weak seeds).
-- R1↔R2 family blend (simplex weights) on validation → apply to test (also tried horizon calibration; kept MSE-only after LB check).

5) Inference
-- CL-aware pad/crop; predict; inverse-scale via preprocessor stats; clip ≥0; write Kaggle submission.

### Challenges
--Per-window normalization vs global scaling: fixed by relying on TTM’s internal scaler + strict dtype control.
--Shape/CL mismatches: solved with left-pad to 512 & observed-masking in collator.
--Seed stability & leakage guardrails: region one-hot fit on train only; early stopping on eval MSE.
--Leaderboard sensitivity: NLL-style calibration helped offline but hurt LB MSE → reverted to MSE-only blending.

### Expected Impact
--Operational: tighter day-ahead forecasts to schedule chillers/shift loads; fewer false alerts (no wild calibration tricks).
--Energy/₹: forecasting accuracy → actionable tariff-weighted scheduling; groundwork for ₹/day & CO₂/day accounting.
--Scalability: small trainable head/decoder; fast few-shot rollout across buildings/regions.

### What we learned / Novelty
--Seed + family blending matters: R1 and R2 learn complementary errors; convex blending beats either alone (e.g., best public ~0.43486 vs single-family ensembles ~0.435–0.436).
--RPT off > on for this dataset/splits (kept off).
--Strict preprocessing hygiene (float32, no object dtypes, leak-safe region dummies) prevents silent degradations.
--Simple beats clever on MSE LB: per-horizon/region calibration that helps NLL didn’t help MSE → removed.

### What’s next
1) Depth of ensemble (low-risk, likely lift):
--Expand to 5 seeds per family (e.g., R1: 40/42/50/17/73; R2: 40/42/44/19/79), convex-blend per family → blend families.
--Keep data_seed=SEED alongside seed to decorrelate loaders.
2) Targeted micro-unfreeze (optional A/B):
--Unfreeze last encoder block for 3–5 epochs with tiny LR (1e-5→5e-5) and ES patience=2–3; stop if no gain.
3) Residual stacking (low-medium risk):
--Train a tiny residual regressor on validation residuals using exogenous features (hour/region) and apply to test. Freeze if public LB drops.
4) Model diversity:
--Add PatchTSMixer (tsfm_public) as a third family for a 3-way blend (R1/R2/PTM).
5) Repro pack:
--One-click script to train seeds → save val/test → run convex blends → emit final CSV.
