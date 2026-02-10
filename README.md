# Diabetes Risk Predictor (XGBoost) — Kaggle PS S5E12

This project builds an end-to-end **machine learning pipeline** to predict `diagnosed_diabetes` (binary classification) using the Kaggle competition:

- **Kaggle**: Playground Series — S5E12  
- **Task**: Supervised **binary classification**
- **Metric**: **ROC AUC**
- **Final Model**: **XGBoost (Gradient Boosting Trees)** trained on a **V1 numeric dataset** (categorical → one-hot encoded)

> **Disclaimer:** This app provides an ML-based risk estimate and is **not** a medical diagnosis.

---

## Results (Kaggle)

- **Random Forest (baseline, V1):** Public 0.68336 | Private 0.67906  
- **XGBoost (final, V1):** **Public 0.69572 | Private 0.69268**

---

## Folder Structure

Place these files in one folder (example: `D:\ML\`):

```
D:\ML\
  app.py
  xgb_v1_model.json
  v1_feature_columns.pkl
  requirements.txt   (optional but recommended)
```

---

## 1) What is the “V1” Dataset?

The original Kaggle data contains **categorical** columns (e.g., `gender`, `smoking_status`).  
To make everything numeric (and to avoid correlation/encoding issues), we create a **V1 dataset**:

- Fill missing values  
  - numeric → median  
  - categorical → most frequent
- One-hot encode categoricals (`pd.get_dummies`)
- Ensure train/test have **identical columns** by combining before encoding
- Save:
  - `train_v1.csv`
  - `test_v1.csv`
  - `y_v1.csv`

The final XGBoost model is trained on this V1 numeric representation.

---

## 2) Export the Final Model from Kaggle (Recommended)

In your Kaggle notebook, after training `xgb_final` (an `XGBClassifier`) on the full V1 dataset:

```python
import joblib

# Save Booster in native JSON format (portable across environments)
xgb_final.get_booster().save_model("xgb_v1_model.json")

# Save the exact feature column order used in training (critical for deployment)
joblib.dump(list(X_full.columns), "v1_feature_columns.pkl")
```

Download these two files from the Kaggle notebook output:

- `xgb_v1_model.json`
- `v1_feature_columns.pkl`

---

## 3) Run the Streamlit App Locally (Windows)

### Step A — Create & activate a virtual environment (recommended)

```powershell
cd D:\ML
python -m venv .venv
python -m pip install --upgrade pip
```

### Step B — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:

```powershell
python -m pip install streamlit pandas numpy joblib xgboost
```

### Step C — Run the app

```powershell
python -m streamlit run app.py
```

Then open the URL shown (usually `http://localhost:8501`).

---

## 4) Deploy on Streamlit Cloud (Optional)

1. Push your project folder to GitHub
2. In Streamlit Cloud, choose the repo and set the entry file to `app.py`
3. Include `requirements.txt` so Streamlit uses compatible versions

---

## requirements.txt (recommended)

Create a file named `requirements.txt` in the same folder:

```
streamlit
pandas
numpy
joblib
xgboost
```

*(If you want strict reproducibility, pin versions, e.g. `xgboost==2.0.3`, but it’s optional when using Booster JSON.)*

---

## 5) Common Errors & Fixes

### ✅ `streamlit is not recognized`
Use:

```powershell
python -m streamlit run app.py
```

### ✅ `ModuleNotFoundError: joblib`
```powershell
python -m pip install joblib
```

### ✅ `ModuleNotFoundError: xgboost`
```powershell
python -m pip install xgboost
```

### ✅ Pickle model fails / version issues
Avoid `.pkl` models for XGBoost. Use:

- `xgb_v1_model.json` (Booster native format)
- `v1_feature_columns.pkl`

This is why the app loads the model using `xgboost.Booster()`.

---

## 6) How Predictions Work in the App

1. User enters patient details (all original features)
2. The app creates one row in the **original schema**
3. Converts it to **V1** using one-hot encoding
4. Aligns the result to the **exact training feature columns**
5. Predicts diabetes probability using the XGBoost Booster

---

## 7) Academic Notes (Report Writing)

- The task is **binary classification** because the output is a class label (0/1)
- ROC AUC is preferred over accuracy when class imbalance exists
- XGBoost was selected because it improved ROC AUC and generalised well (public/private scores close)

Suggested wording:

> “XGBoost was selected as the final model due to its improved ROC AUC performance and stable generalisation on Kaggle public and private evaluation sets. The deployment package includes the trained booster in JSON format and the exact one-hot feature schema to ensure consistent predictions in the application environment.”

---

## License / Usage

For educational use as part of coursework and learning on Kaggle.
