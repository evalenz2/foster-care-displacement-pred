import streamlit as st
import pandas as pd
import joblib
import json
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load model and threshold
model = joblib.load("foster_model.pkl")
with open("model_threshold.json") as f:
    threshold = json.load(f)['threshold']
with open("feature_names.json") as f:
    feature_names = json.load(f)

# Load SHAP explainer
explainer = joblib.load("shap_explainer.pkl")

# Categorical options mapping (expanded for clarity in UI)
categorical_options = {
    "SEX": {"Male": 1, "Female": 2},
    "RaceEthn": {
        "White": 1,
        "Black or African American": 2,
        "Hispanic": 3,
        "American Indian or Alaska Native": 4,
        "Asian": 5,
        "Two or More Races": 6,
        "Unknown/Other": 7
    },
    "HISORGIN": {"No": 0, "Yes": 1},
    "PHYABUSE": {"No": 0, "Yes": 1},
    "SEXABUSE": {"No": 0, "Yes": 1},
    "NEGLECT": {"No": 0, "Yes": 1},
    "CHBEHPRB": {"No": 0, "Yes": 1},
    "EmotDist": {"No": 0, "Yes": 1},
    "PRTSJAIL": {"No": 0, "Yes": 1},
    "NOCOPE": {"No": 0, "Yes": 1},
    "ABANDMNT": {"No": 0, "Yes": 1},
    "RELINQSH": {"No": 0, "Yes": 1},
    "HOUSING": {"No": 0, "Yes": 1},
    "AAPARENT": {"No": 0, "Yes": 1, "Yes, Adoptive Mother Present": 2},
    "DAPARENT": {"No": 0, "Yes": 1, "Yes, Adoptive Father Present": 2},
    "AACHILD": {"No": 0, "Yes": 1},
    "DACHILD": {"No": 0, "Yes": 1},
    "CHILDIS": {"No": 0, "Yes": 1},
    "CLINDIS": {"No": 0, "Yes": 1},
    "MR": {"No": 0, "Yes": 1},
    "VISHEAR": {"No": 0, "Yes": 1},
    "PHYDIS": {"No": 0, "Yes": 1},
    "OTHERMED": {"No": 0, "Yes": 1},
    "PRTSDIED": {"No": 0, "Yes": 1},
    "IsWaiting": {"No": 0, "Yes": 1, "Waiting Category 2": 2},
    "IVEFC": {"No": 0, "Yes": 1},
    "IVAAFDC": {"No": 0, "Yes": 1},
    "XIXMEDCD": {"No": 0, "Yes": 1},
    "PLACEOUT": {"No": 0, "Yes": 1},
    "IsTPR": {"No": 0, "Yes": 1},
    "CURPLSET": {
        "Pre-Adoptive Home": 1,
        "Foster Family Home (Relative)": 2,
        "Foster Family Home (Non-Relative)": 3,
        "Group Home": 4,
        "Institution": 5,
        "Supervised Independent Living": 6,
        "Runaway": 7,
        "Trial Home Visit": 8,
        "Other": 9
    },
    "CASEGOAL": {
        "Reunify with Parent(s)/Caretaker(s)": 1,
        "Live with Other Relatives": 2,
        "Adoption": 3,
        "Long-Term Foster Care": 4,
        "Emancipation": 5,
        "Guardianship": 6,
        "Case Plan Goal Not Yet Established": 7
    }
}

# Translated labels for form display
field_labels = {
    "SEX": "Sex",
    "AgeAtStart": "Age at Start",
    "HISORGIN": "Hispanic Origin",
    "RaceEthn": "Race/Ethnicity",
    "PHYABUSE": "Physical Abuse",
    "SEXABUSE": "Sexual Abuse",
    "NEGLECT": "Neglect",
    "AAPARENT": "Adoptive Mother Present",
    "DAPARENT": "Adoptive Father Present",
    "AACHILD": "Child Asked for Adoption",
    "DACHILD": "Child Denied Adoption",
    "CHILDIS": "Child Disability",
    "CHBEHPRB": "Behavioral Problems",
    "EmotDist": "Emotional Disturbance",
    "CLINDIS": "Clinically Diagnosed Disability",
    "MR": "Mental Retardation",
    "VISHEAR": "Visual/Hearing Impairment",
    "PHYDIS": "Physical Disability",
    "OTHERMED": "Other Medical Condition",
    "PRTSDIED": "Parent(s) Deceased",
    "PRTSJAIL": "Parent(s) Incarcerated",
    "NOCOPE": "Parents Unable to Cope",
    "ABANDMNT": "Abandonment",
    "RELINQSH": "Relinquishment",
    "HOUSING": "Inadequate Housing",
    "CASEGOAL": "Case Goal",
    "CURPLSET": "Current Placement Setting",
    "PLACEOUT": "Placed Out of Home",
    "IsTPR": "Termination of Parental Rights",
    "IsWaiting": "Is Waiting for Adoption",
    "IVEFC": "IV-E Foster Care Eligible",
    "IVAAFDC": "IV-A/AFDC Eligible",
    "XIXMEDCD": "Medicaid Eligible"
}

st.set_page_config(page_title="Foster Care Displacement Predictor", layout="centered")
st.title("🧒 Foster Care Displacement Risk Predictor")

st.markdown("""
This tool uses a machine learning model trained on historical foster care data to help predict the likelihood of a child experiencing multiple placements (displacement) in the foster care system. 

**What does the prediction mean?**
- The model outputs a probability between 0 and 1.
- If that probability is **greater than 0.32**, the model predicts the child is at risk of displacement.
- This tool is designed to help social workers and case managers flag potentially unstable placements early.

The model prioritizes **recall**, meaning it aims to catch as many true displacement cases as possible—even if it means some false alarms.
""")

st.markdown("Fill in the child-specific information below to estimate the risk of future displacement.")

# User input form
input_data = {}
with st.form("input_form"):
    for feature in feature_names:
        label = field_labels.get(feature, feature)
        if feature in categorical_options:
            options = list(categorical_options[feature].keys())
            selected = st.selectbox(label, options)
            input_data[feature] = categorical_options[feature][selected]
        else:
            input_data[feature] = st.number_input(label, value=0)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Create DataFrame
        user_df = pd.DataFrame([input_data])

        # Predict
        prob = model.predict_proba(user_df)[:, 1][0]
        pred = int(prob >= threshold)

        st.subheader("Prediction Result:")
        st.write(f"**Probability of displacement:** {prob:.2%}")
        st.write(f"**Prediction (Threshold = {threshold}):** {'Displaced' if pred else 'Stable'}")

        # SHAP Explanation
        st.subheader("Explanation of Prediction:")
        shap_values = explainer(user_df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
