import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ================================
# LOAD MODELS & ARTIFACTS
# ================================
clf = joblib.load("../models/xgb_classifier.pkl")
reg = joblib.load("../models/xgb_regressor.pkl")
columns = joblib.load("../models/feature_columns.pkl")
medians = joblib.load("../models/median_values.pkl")

st.set_page_config(page_title="EMI Predictor", layout="wide")

st.title("💰 EMI Prediction & Financial Risk Dashboard")

# ================================
# USER INPUT UI
# ================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")

    age = st.slider("Age", 18, 60, int(medians['age']))
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    company_type = st.selectbox("Company Type", ["Large Indian", "MNC", "Mid-size", "Startup", "Small"])

with col2:
    st.subheader("💰 Financial Details")

    salary = st.number_input("Monthly Salary (₹)", 10000, 200000, int(medians['monthly_salary']))
    bank_balance = st.number_input("Bank Balance (₹)", 0, 1000000, int(medians['bank_balance']))
    emergency_fund = st.number_input("Emergency Fund (₹)", 0, 500000, int(medians['emergency_fund']))
    credit_score = st.slider("Credit Score", 300, 850, int(medians['credit_score']))
    existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])

st.subheader("🏠 Lifestyle & Expenses")

col3, col4 = st.columns(2)

with col3:
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    monthly_rent = st.number_input("Monthly Rent (₹)", 0, 50000, int(medians['monthly_rent']))
    groceries = st.number_input("Groceries & Utilities (₹)", 0, 50000, int(medians['groceries_utilities']))
    travel = st.number_input("Travel Expenses (₹)", 0, 30000, int(medians['travel_expenses']))
    other_exp = st.number_input("Other Expenses (₹)", 0, 50000, int(medians['other_monthly_expenses']))

with col4:
    school_fees = st.number_input("School Fees (₹)", 0, 50000, int(medians['school_fees']))
    college_fees = st.number_input("College Fees (₹)", 0, 50000, int(medians['college_fees']))
    family_size = st.number_input("Family Size", 1, 10, int(medians['family_size']))
    dependents = st.number_input("Dependents", 0, 10, int(medians['dependents']))
    current_emi = st.number_input("Current EMI (₹)", 0, 50000, int(medians['current_emi_amount']))

st.subheader("🏦 Loan Request")

col5, col6 = st.columns(2)

with col5:
    emi_scenario = st.selectbox("EMI Scenario", [
        "Home Appliances EMI",
        "Personal Loan EMI",
        "E-commerce Shopping EMI",
        "Education EMI",
        "Vehicle EMI"
    ])
    requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 2000000, int(medians['requested_amount']))

with col6:
    tenure = st.number_input("Tenure (months)", 3, 120, int(medians['requested_tenure']))
    years_employment = st.number_input("Years of Employment", 0, 40, int(medians['years_of_employment']))

# ================================
# NORMALIZE
# ================================
gender = gender.lower()
education = education.lower()
employment_type = employment_type.lower()
company_type = company_type.lower()
house_type = house_type.lower()
emi_scenario = emi_scenario.lower()

# ================================
# PREPARE INPUT
# ================================
input_dict = medians.to_dict()

input_dict.update({
    'age': age,
    'monthly_salary': salary,
    'credit_score': credit_score,
    'existing_loans': 1 if existing_loans == "Yes" else 0,
    'marital_status': 1 if marital_status == "Married" else 0,
    'bank_balance': bank_balance,
    'emergency_fund': emergency_fund,
    'monthly_rent': monthly_rent,
    'groceries_utilities': groceries,
    'travel_expenses': travel,
    'other_monthly_expenses': other_exp,
    'school_fees': school_fees,
    'college_fees': college_fees,
    'family_size': family_size,
    'dependents': dependents,
    'current_emi_amount': current_emi,
    'requested_amount': requested_amount,
    'requested_tenure': tenure,
    'years_of_employment': years_employment
})

input_df = pd.DataFrame([input_dict])

# One-hot encoding
input_df['gender'] = gender
input_df['education'] = education
input_df['employment_type'] = employment_type
input_df['company_type'] = company_type
input_df['house_type'] = house_type
input_df['emi_scenario'] = emi_scenario

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

# ================================
# PREDICTION
# ================================
if st.button("🚀 Predict"):

    eligibility = clf.predict(input_df)[0]
    emi = reg.predict(input_df)[0]
    emi = max(0, emi)

    label_map = {
        0: "Eligible",
        1: "High Risk",
        2: "Not Eligible"
    }

    result = label_map.get(eligibility)

    # ================================
    # BUSINESS RULES (CRITICAL)
    # ================================
    total_expenses = (
        monthly_rent + groceries + travel + other_exp +
        school_fees + college_fees + current_emi
    )

    remaining_income = salary - total_expenses

    if remaining_income <= 0:
        result = "Not Eligible"
        emi = 0

    max_allowed_emi = salary * 0.4
    emi = min(emi, max_allowed_emi)

    # ================================
    # DISPLAY RESULTS
    # ================================
    st.subheader("📊 Results")

    if result == "Not Eligible":
        st.error("❌ Not Eligible")
    elif result == "High Risk":
        st.warning("⚠️ High Risk")
    else:
        st.success("✅ Eligible")

    st.metric("💵 Maximum EMI", f"₹ {int(emi):,}")

    # ================================
    # CHART: INCOME VS EXPENSES
    # ================================
    st.subheader("📊 Income vs Expenses")

    fig, ax = plt.subplots()
    ax.bar(["Income", "Expenses"], [salary, total_expenses])
    st.pyplot(fig)

    # ================================
    # RISK GAUGE
    # ================================
    st.subheader("🧭 Risk Level")

    if result == "Eligible":
        st.progress(20)
        st.success("Low Risk")
    elif result == "High Risk":
        st.progress(60)
        st.warning("Moderate Risk")
    else:
        st.progress(90)
        st.error("High Risk")

    # ================================
    # EMI BREAKDOWN
    # ================================
    st.subheader("💵 EMI Breakdown")

    breakdown = pd.DataFrame({
        "Category": ["Income", "Expenses", "Remaining", "Recommended EMI"],
        "Amount": [salary, total_expenses, remaining_income, emi]
    })

    st.table(breakdown)

    st.info("EMI is calculated based on disposable income after expenses.")