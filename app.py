import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
input_metadata = joblib.load("input_metadata.pkl")

st.set_page_config(
    page_title="Biology Infection Risk Prediction",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Biology Infection Risk Prediction")
st.write("Klasifikasi Infection Risk Level menggunakan Random Forest.")

raw_features = input_metadata["raw_feature_names"]
numeric_features = input_metadata["numeric_features"]
ordinal_features = input_metadata["ordinal_features"]
nominal_features = input_metadata["nominal_features"]
ordinal_categories = input_metadata["ordinal_categories"]
nominal_categories = input_metadata["nominal_categories"]

st.subheader("Input Data")

input_data = {}

for feature in raw_features:

    if feature in ordinal_features:
        input_data[feature] = st.selectbox(
            feature,
            ordinal_categories[feature]
        )

    elif feature in nominal_features:
        input_data[feature] = st.selectbox(
            feature,
            nominal_categories[feature]
        )

    else:
        input_data[feature] = st.number_input(
            feature,
            value=0.0
        )

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    processed_input = preprocessor.transform(input_df)

    prediction = model.predict(processed_input)[0]

    st.success(
        f"Predicted Infection Risk Level: {prediction}"
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_input)[0]

        probability_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": probabilities
        })

        st.subheader("Prediction Probability")
        st.dataframe(probability_df)
