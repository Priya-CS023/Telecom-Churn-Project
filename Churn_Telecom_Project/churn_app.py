import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load model and threshold
# -----------------------------

model = joblib.load("models/churn_model.pkl")
threshold = joblib.load("models/threshold.pkl")


# -----------------------------
# Page
# -----------------------------

st.set_page_config(
    page_title="Churn Prediction"
)

st.title("Telecom Churn Prediction App")
#st.write("Predict whether a customer is likely to churn.")


# -----------------------------
# Customer Information
# -----------------------------

st.header("Customer Information")

col1, col2 = st.columns(2)


# -----------------------------
# Column 1
# -----------------------------

with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No"]
    )


# -----------------------------
# Column 2
# -----------------------------

with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    


# -----------------------------
# Create Customer DataFrame
# -----------------------------

customer = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "SeniorCitizen": [senior_citizen],

    "gender": [gender],
    "Partner": [partner],
    "Dependents": [dependents],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaymentMethod": [payment_method],
})


# -----------------------------
# Prediction Button
# -----------------------------

st.write("")

if st.button("Predict"):

    # -------------------------
    # Probability
    # -------------------------

    probability = model.predict_proba(customer)[0, 1]

    prediction = int(probability >= threshold)

    no_churn_probability = 1 - probability


    # -------------------------
    # Prediction Result
    # -------------------------

    st.header("Prediction")

    st.write(
        f"Churn probability: **{probability:.1%}**"
    )

    if prediction == 1:

        st.error(
            "Customer is likely to churn"
        )

    else:

        st.success(
            "Customer is unlikely to churn"
        )


    # -------------------------
    # Probability Graph
    # -------------------------

    st.subheader("Churn Probability")

    probability_df = pd.DataFrame({
        "Outcome": ["No Churn", "Churn"],
        "Probability": [
            no_churn_probability,
            probability
        ]
    })

    st.bar_chart(
        probability_df.set_index("Outcome")
    )


    # -------------------------
    # Main Reason
    # -------------------------

    if prediction == 1:

        reasons = []

        if contract == "Month-to-month":

            reasons.append(
                (
                    "Month-to-month contract",
                    "The customer is on a month-to-month contract."
                )
            )

        if tenure < 12:

            reasons.append(
                (
                    "Short tenure",
                    "The customer has relatively short tenure."
                )
            )

        if monthly_charges > 65:

            reasons.append(
                (
                    "High monthly charges",
                    "The customer's monthly charges are relatively high."
                )
            )

        if online_security == "No":

            reasons.append(
                (
                    "No online security",
                    "The customer does not have online security."
                )
            )

        if tech_support == "No":

            reasons.append(
                (
                    "No tech support",
                    "The customer does not have technical support."
                )
            )

        if payment_method == "Electronic check":

            reasons.append(
                (
                    "Electronic check",
                    "The customer uses electronic check as the payment method."
                )
            )


        if reasons:

            reason_title, reason_text = reasons[0]

            st.subheader("Main Reason")

            st.warning(
                f"**{reason_title}**\n\n"
                f"{reason_text}"
            )

        else:

            st.info(
                "The model predicts churn, but no single predefined "
                "reason was identified."
            )