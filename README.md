# 💰 EMI Prediction & Financial Risk Assessment System

An end-to-end Machine Learning project that predicts **loan eligibility** and **maximum affordable EMI** based on a user’s financial profile.

This project combines:
- 📊 Data Analysis & Feature Engineering
- 🤖 Machine Learning (Classification + Regression)
- 📈 Model Tracking with MLflow
- 🌐 Interactive Web App using Streamlit
- 🧠 Business Logic for real-world financial decisions

---

## Dataset: https://drive.google.com/drive/folders/1eCN-NmpxNWVDuT_LY70JPVC6AzDBzWSs?usp=drive_link

## 🎯 Problem Statement

Financial institutions need to assess:
- Whether a user is eligible for EMI
- How much EMI they can safely afford

This project builds a system that:
- Classifies users into:
  - ✅ Eligible
  - ⚠️ High Risk
  - ❌ Not Eligible
- Predicts maximum EMI amount

---

## 🧠 Models Used

### 🔹 Classification
- Logistic Regression (baseline)
- Random Forest
- **XGBoost (Final Model)**

### 🔹 Regression
- Linear Regression
- Random Forest Regressor
- **XGBoost Regressor (Final Model)**

---

## 📊 Model Performance

### Regression (Final Results)

| Model | RMSE | R² |
|------|------|----|
| Linear Regression | 4178 | 0.71 |
| Random Forest | 1383 | 0.96 |
| **XGBoost** | **1037** | **0.98** |

---

### Classification Insights

- Dataset is highly imbalanced
- Techniques used:
  - SMOTE (oversampling)
  - Class balancing
- Focus metric: **Recall for High Risk**

---

## 🧩 Features Used

- Demographics (age, gender, marital status)
- Financial (salary, bank balance, credit score)
- Expenses (rent, groceries, travel, EMI)
- Loan details (amount, tenure, scenario)

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- MLflow
- Streamlit
- Matplotlib

---

---

## 🌐 Streamlit App Features

- 📊 Income vs Expense Visualization
- 🧭 Risk Indicator Gauge
- 💵 EMI Breakdown Table
- 🧠 ML + Business Logic Integration

---

## 🧠 Business Logic Layer

To ensure realistic outputs:
- EMI capped at **40% of income**
- Negative disposable income → **Not Eligible**
- Prevents unrealistic predictions

---

## ▶️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/emi-prediction.git
cd emi-prediction
