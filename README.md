# Diabetes Prediction from Health Indicators (CDC BRFSS 2015)

A Machine Learning project that predicts whether a person has **diabetes or prediabetes** based on health, lifestyle, and socioeconomic indicators from the CDC's **BRFSS 2015** survey (Behavioral Risk Factor Surveillance System).

The main goal is to **detect as many positive cases as possible (high recall)**: in a medical screening context, missing a person with diabetes is more costly than raising a false alarm.

---

## Datasets

Source: [CDC Diabetes Health Indicators – UCI ML Repository (id 891)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) / [Kaggle – Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

| File | Rows | Target | Description |
|---|---|---|---|
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253,680 | `Diabetes_binary` (0/1) | Full dataset, imbalanced (~14% with diabetes) |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70,692 | `Diabetes_binary` (0/1) | Balanced 50/50 (35,346 per class) |

A third version with three classes (no diabetes / prediabetes / diabetes), `diabetes_012_health_indicators_BRFSS2015.csv`, is available from the same Kaggle source but is not used in this project.

**21 predictor variables**, all numeric with no missing values:

- **Clinical health:** `HighBP`, `HighChol`, `CholCheck`, `BMI`, `Stroke`, `HeartDiseaseorAttack`, `DiffWalk`
- **Self-reported health:** `GenHlth` (1–5), `MentHlth` and `PhysHlth` (bad days in the last 30)
- **Lifestyle:** `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`
- **Healthcare access:** `AnyHealthcare`, `NoDocbcCost`
- **Demographic / socioeconomic:** `Sex`, `Age` (13 brackets), `Education`, `Income`

---

## Project structure

```
├── overview.ipynb              # Automatic overview: shape, dtypes, describe
├── 50_50model.ipynb            # EDA + feature engineering + models (balanced 50/50 dataset)
├── results.json                # Final metrics, confusion matrices, and ROC curves (both datasets)
├── pyproject.toml              # Dependencies (managed with uv)
└── *.csv                       # Datasets
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution of each variable split by class (with / without diabetes), using a colorblind-safe palette.
- Correlation matrix and ranking of variables by their correlation with the target.

### 2. Feature engineering
New features were built on clinical reasoning:

| Feature | Definition | Rationale |
|---|---|---|
| `BMI_category` | Underweight / Normal / Overweight / Obese (one-hot, `drop_first`) | Clinical BMI ranges, helps linear models with non-linearity |
| `UnhealthyDays` | MentHlth + PhysHlth (capped at 30) | Total bad days per month |
| `risk_score` | HighBP + HighChol + Stroke + HeartDisease + DiffWalk | Accumulated comorbidities (most correlated feature, r ≈ 0.45) |
| `MetSyn_count` | HighBP + HighChol + (BMI ≥ 30) | Metabolic syndrome proxy |
| `Age_45plus` | Age ≥ 45 years | ADA screening cut-off |
| `FreqPhysDistress` | PhysHlth ≥ 14 | CDC definition of "frequent distress" |
| `PoorHealth` | GenHlth ≥ 4 | Fair or poor self-rated health |
| `CVD` | Stroke or HeartDisease | Merges two rare events into a stronger signal |
| `SevereObese` | BMI ≥ 40 (BMI capped at 60) | Sharply higher risk and outlier control |
| `Sedentary_Obese` | No physical activity and BMI ≥ 30 | Target group of diabetes prevention programs |
| `Lifestyle_score` | Activity + fruits + veggies + non-smoker + no heavy drinking | Combines individually weak signals |
| `SES` | Income/8 + Education/6 | Socioeconomic index |
| Interactions | `BMI×HighBP`, `BMI×HighChol`, `Age×BMI`, `Age×HighBP`, `Smoker×Age`, `Smoker×HeartDisease` | Combined effects |

### 3. Feature selection
1. **Correlation:** dropped variables with |r| < 0.1 against the target (`Fruits`, `Veggies`, `Smoker`, `MentHlth`, `AnyHealthcare`, ...). `Sex` was kept because the literature shows it matters.
2. **VIF (Variance Inflation Factor):** dropped redundant variables and exact combinations of other variables (`risk_score`, `SES`, BMI interactions, `Age_BMI`, ...) to reduce multicollinearity.
3. **With vs. without `MetSyn_count`**, compared through cross-validation. Results were practically identical, so the simpler version **without** it was chosen (20 final features).

### 4. Modeling
- `StandardScaler` + classifier pipeline, 75/25 train/test split (`random_state=42`).
- **5-fold cross-validation** on ROC-AUC, recall, precision, and F1 to compare Logistic Regression, Decision Tree, Random Forest, KNN, Gradient Boosting, and SVC.

  | Model | ROC-AUC | Recall | Precision | F1 |
  |---|---|---|---|---|
  | Gradient Boosting | 0.829 | 0.793 | 0.731 | 0.761 |
  | Logistic Regression | 0.825 | 0.771 | 0.738 | 0.754 |
  | SVC | 0.809 | 0.792 | 0.728 | 0.759 |
  | Random Forest | 0.802 | 0.767 | 0.717 | 0.741 |
  | KNN | 0.771 | 0.738 | 0.701 | 0.719 |
  | Decision Tree | 0.656 | 0.647 | 0.659 | 0.653 |

  *5-fold CV on the balanced training set, without `MetSyn_count`.*

- **Selected models:** Logistic Regression and Gradient Boosting.
- **GridSearchCV** hyperparameter tuning:
  - Logistic Regression: `C = 0.01`
  - Gradient Boosting: `n_estimators = 200`, `learning_rate = 0.1`, `max_depth = 3`
- **Decision threshold tuning** to prioritize recall.

---

## Results (test set)

### Balanced 50/50 dataset (70,692 rows)

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.825 | 0.769 | 0.736 | 0.752 |
| **Gradient Boosting** | **0.830** | **0.798** | 0.731 | **0.763** |

### Full dataset (253,680 rows, ~14% positive)

Same pipeline applied to the imbalanced dataset. The metrics are stored in `results.json`.

| Model | ROC-AUC | Recall (threshold 0.5) | Recall (tuned threshold) | Precision (tuned) | F1 (tuned) |
|---|---|---|---|---|---|
| Logistic Regression | 0.826 | 0.162 | 0.804 (threshold 0.13) | 0.298 | 0.435 |
| **Gradient Boosting** | **0.832** | 0.157 | 0.793 (threshold 0.14) | 0.309 | **0.445** |

### Key takeaways
- **Gradient Boosting** is the best model in both settings, though Logistic Regression is very close and easier to interpret.
- **ROC-AUC is nearly identical (~0.83)** on both datasets. The difference is the decision threshold: on imbalanced data, the default 0.5 threshold catches only ~16% of cases.
- **Lowering the threshold to ~0.13–0.14** on the full dataset raises recall from 16% to ~80%, at the cost of more false positives (precision ~30%). For screening, this is a reasonable trade-off.
- The strongest predictors are **accumulated comorbidities** (high blood pressure, high cholesterol, heart disease), **self-rated general health**, **BMI**, and **age**.

---

## Installation and usage

Requires **Python ≥ 3.12**. The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/Gaby-projects/Machine_Learning_Models_Diabetes_Prediction.git
cd Machine_Learning_Models_Diabetes_Prediction

uv sync          # creates the virtual environment and installs dependencies
```

Then open the notebooks in Jupyter or VS Code with the `.venv` kernel. Suggested order:

1. `overview.ipynb`
2. `50_50model.ipynb`

**Main libraries:** pandas, scikit-learn, xgboost, statsmodels, seaborn, matplotlib, ucimlrepo.

