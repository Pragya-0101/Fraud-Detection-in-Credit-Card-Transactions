import streamlit as st
import pandas as pd
import joblib


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("fraud_model.pkl")
amount_transformer = joblib.load("amount_transformer.pkl")
scaler = joblib.load("scaler.pkl")


# ==============================
# APP
# ==============================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Upload a transaction CSV to predict whether transactions "
    "are legitimate or fraudulent."
)

st.info(
    "The CSV must contain Time, V1–V28, and Amount."
)


# ==============================
# FILE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ==============================
    # REQUIRED COLUMNS
    # ==============================

    required_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    else:

        st.success("CSV uploaded successfully.")

        st.subheader("Transaction Data")

        st.dataframe(df.head())


        # ==============================
        # MODEL INPUT
        # ==============================

        X = df[required_columns].copy()


        # ==============================
        # PREPROCESSING
        # ==============================

        # Convert Time from seconds to hours
        X["Time"] = X["Time"] / 3600


        # Yeo-Johnson transformation
        X["Amount"] = amount_transformer.transform(
            X[["Amount"]]
        )


        # Scale Time and Amount
        X[["Time", "Amount"]] = scaler.transform(
            X[["Time", "Amount"]]
        )


        # ==============================
        # PREDICTION
        # ==============================

        if st.button("Predict Fraud"):

            predictions = model.predict(X)

            probabilities = model.predict_proba(X)[:, 1]


            # ==============================
            # RESULTS
            # ==============================

            results = df.copy()

            results["Fraud Probability"] = probabilities

            results["Prediction"] = [
                "Fraudulent" if p == 1
                else "Legitimate"
                for p in predictions
            ]


            st.subheader("Prediction Results")

            st.dataframe(
                results[
                    [
                        "Time",
                        "Amount",
                        "Fraud Probability",
                        "Prediction"
                    ]
                ]
            )


            # ==============================
            # SUMMARY
            # ==============================

            total = len(results)

            fraud = sum(predictions == 1)

            legitimate = total - fraud


            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Transactions",
                total
            )

            col2.metric(
                "Legitimate",
                legitimate
            )

            col3.metric(
                "Fraudulent",
                fraud
            )


            # ==============================
            # DOWNLOAD
            # ==============================

            output = results.to_csv(index=False)

            st.download_button(
                "Download Predictions",
                output,
                "fraud_predictions.csv",
                "text/csv"
            )