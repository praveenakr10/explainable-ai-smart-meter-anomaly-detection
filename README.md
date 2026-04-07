# XAI Smart Meter - Explainable AI for Interpreting Smart Meter Anomalies

 Detecting electricity consumption anomalies in smart meter data using Machine Learning, Deep Learning, and a full suite of Explainable AI (XAI) techniques.

---

##  Overview

Smart meters generate continuous electricity readings that need to be monitored for anomalies, signs of theft, equipment faults, or unusual usage. Standard ML models can detect these anomalies but offer no explanation for their decisions, making them difficult to trust or act on.

This project builds **an end-to-end XAI pipeline** that is both highly accurate and fully interpretable. It combines classical ML, a Hybrid BiLSTM + Attention deep learning model, and four complementary XAI methods, evaluated for faithfulness, stability, and agreement.

---

## Dataset

| Attribute | Details |
|---|---|
| Source | [Kaggle – Smart Meter Electricity Consumption (ziya07)](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset) |
| Records | 5,000 half-hourly readings |
| Raw Features | Timestamp, Electricity_Consumed, Temperature, Humidity, Wind_Speed, Avg_Past_Consumption, Anomaly_Label |
| Engineered Features | 14 total (hour, day, weekday, month, Monthly_Mean, Monthly_STD, Day_Part OHE) |
| Class Distribution | 4,750 Normal (95%) · 250 Anomaly (5%) - severe imbalance |
| Imbalance Fix | SMOTETomek to 3,799 balanced per class in training set |

---

## Models Trained

### Classical ML (7 models)

| Model | Precision | Recall | F1 (Anomaly) | Accuracy |
|---|---|---|---|---|
| **XGBoost** | 0.73 | **0.98** | **0.84** | 98% |
| **Random Forest** | 0.73 | 0.76 | 0.75 | 97% |
| SVM | 0.55 | 0.88 | 0.68 | 96% |
| Decision Tree | 0.53 | 0.90 | 0.67 | 95% |
| KNN | 0.36 | 0.60 | 0.45 | 93% |
| Naive Bayes | 0.16 | 0.82 | 0.27 | 78% |
| Logistic Regression | 0.04 | 0.36 | 0.08 | 57% |

> Random Forest selected as primary XAI model (ROC-AUC: **0.9865**)

### Hybrid Deep Learning Model

```
Input (24 timesteps × 14 features)
        ↓
Bidirectional LSTM (64 units, return_sequences=True)
        ↓
Custom Attention Layer (softmax-weighted timestep importance)
        ↓
Dense (32 units, ReLU)  →  32-dim learned embeddings
        ↓
Dropout (0.3)
        ↓
Random Forest Classifier (100 estimators)
```

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.95 | 1.00 | **0.97** | 755 |
| Anomaly | 1.00 | 0.95 | **0.97** | 760 |
| **Overall Accuracy** | | | **97%** | 1515 |

---

## XAI Methods

### 1. SHAP: SHapley Additive Explanations
- **Global** beeswarm summary plot — all 1,000 test instances ranked by mean |SHAP|
- **Local** waterfall plot: individual prediction decomposition (baseline to final probability)
- **Dependence plot**: non-linear threshold effect of `Electricity_Consumed`
- Applied to both **Random Forest** and **Hybrid model** (embedding space)

### 2. LIME: Local Interpretable Model-agnostic Explanations
- Instance-level **contrastive rules** (e.g. `-0.08 < Electricity_Consumed <= 1.09`)
- Applied to **RF** (original 14 features) and **Hybrid RF** (32 learned embeddings)
- **Stability analysis** across 10 similar anomaly instances

### 3. PDP + ICE: Partial Dependence & Individual Conditional Expectation
- **PDP** for top-4 features: average marginal effect on anomaly probability
- **ICE** for top-2 features: individual effect per test instance revealing heterogeneity

### 4. DiCE: Diverse Counterfactual Explanations
- Generates 5 diverse **minimum-change scenarios** to flip Anomaly to Normal
- Answers: *"What would need to change for this not to be an anomaly?"*
- Feature delta heatmap for all 5 counterfactuals

---

## XAI Evaluation Results

| Metric | Result | Interpretation |
|---|---|---|
| **Faithfulness** (AUC drop after ablation) | **0.4764** | Removing top SHAP features causes 47.6% AUC collapse → SHAP correctly identifies what matters |
| **Stability** (mean SHAP Std) | **0.0475** | Low variance across similar instances → consistent, reliable explanations |
| **SHAP vs Permutation** (Spearman ρ) | **0.934** | Very strong agreement |
| **LIME vs Permutation** (Spearman ρ) | **0.925** | Very strong agreement |
| **SHAP vs LIME** (Spearman ρ) | **0.851** | Strong agreement |

### Top Feature Consensus (all 3 methods agree)

| Rank | Feature | SHAP | LIME | Permutation |
|---|---|---|---|---|
| 1 | Avg_Past_Consumption | #1 | #2 | #1 |
| 2 | Electricity_Consumed | #2 | #1 | #2 |
| 3 | Monthly_STD | #3 | #5 | #4 |
| 4 | Monthly_Mean | #4 | #3 | #3 |

---

## Getting Started

### 1. Clone the repo

### 2. Install dependencies

### 3. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset) and place at:
```
/kaggle/input/datasets/ziya07/smart-meter-electricity-consumption-dataset/smart_meter_data.csv
```

### 4. Run the notebook

```bash
jupyter notebook xai-smart-meter-final.ipynb
```

---

## Key Findings

- **`Avg_Past_Consumption` is more important than current consumption**, the model detects anomalies relative to a meter's own history, not absolute thresholds. This is a personalised, context-aware detection approach.

- **Seasonal context is critical**, `Monthly_Mean` and `Monthly_STD` rank 3rd-4th consistently. The same consumption level can be normal in winter but anomalous in summer.

- **BiLSTM+Attention dramatically improves anomaly detection**, from F1 0.75 (plain RF) to F1 0.97 by capturing 24-hour temporal patterns that point-in-time models miss.

- **XAI methods are consistent**, Spearman ρ > 0.85 across all method pairs confirms explanations are genuine reflections of model behaviour, not method artefacts.

- **Counterfactuals reveal seasonal context as the primary flip mechanism**, in 5/5 counterfactuals, shifting the monthly context was part of the minimum-change path from Anomaly to Normal.

---

## 🧠 Architecture Diagram

```
Raw Data (5000 × 7)
        │
        ▼
Feature Engineering (→ 14 features)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
Classical ML Models              BiLSTM + Attention
(RF, XGBoost, SVM, DT,           24-step sequences
 KNN, NB, LR)                    → 32-dim embeddings
        │                                  │
        ▼                                  ▼
   XAI: SHAP                     Hybrid RF Classifier
   XAI: LIME                          │
   XAI: PDP/ICE                       ▼
   XAI: DiCE              XAI: SHAP (embeddings)
        │                 XAI: LIME (embeddings)
        └──────────┬───────────────────────┘
                   ▼
        XAI Evaluation Framework
        (Faithfulness · Stability · Agreement)
```

---

## Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Data Loading & EDA | Shape, nulls, duplicates, class distribution |
| 2 | Feature Engineering | Temporal features, seasonal stats, Day_Part OHE |
| 3 | Preprocessing | Train/test split, StandardScaler, SMOTETomek |
| 4 | Classical ML | 7 models trained and compared |
| 5 | SHAP | Beeswarm, waterfall, dependence plots |
| 6 | BiLSTM + Attention | Sequence creation, model architecture, training |
| 7 | Hybrid RF | Embedding extraction, RF on embeddings |
| 8 | SHAP (Hybrid) | SHAP on embedding space |
| 9 | LIME | RF + Hybrid explanations + stability |
| 10 | PDP + ICE | Partial dependence and individual effects |
| 11 | DiCE | Counterfactual generation + heatmap |
| 12 | XAI Evaluation | Faithfulness, stability, inter-method agreement |

## Conclusion

This project demonstrates how Explainable AI (XAI) techniques can be used to make machine learning models more transparent and trustworthy in smart meter anomaly detection. By combining performance with interpretability, the system provides meaningful insights into model decisions.
