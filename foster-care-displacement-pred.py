import streamlit as st
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import pandas as pd

# Initialize H2O
h2o.init()

# Load the trained model
model_path = "./Grid_GBM_py_11_sid_8652_model_python_1744321074497_443_model_5"  # Update if your repo structure changes
model = h2o.load_model(model_path)

# Define features (ensure the order matches training)
features = [
    'sex', 'ageatstart', 'phyabuse', 'sexabuse', 'neglect', 'chbehprb',
    'prtsjail', 'nocope', 'abandmnt', 'relinqsh', 'housing', 'curplset',
    'placeout', 'casegoal', 'raceethn', 'hisorgin', 'emotdist'
]

st.title("Foster Care Displacement Risk Predictor")
st.write("Enter the child's information to predict risk of displacement.")

# User inputs
input_data = {}
for feat in features:
    if feat in ['sex', 'phyabuse', 'sexabuse', 'neglect', 'chbehprb', 'prtsjail', 'nocope', 'abandmnt', 'relinqsh', 'housing', 'emotdist']:
        input_data[feat] = st.selectbox(f"{feat}", options=[0, 1])
    elif feat in ['curplset', 'placeout', 'casegoal', 'raceethn']:
        input_data[feat] = st.number_input(f"{feat} (code)", min_value=0, step=1)
    elif feat == 'hisorgin':
        input_data[feat] = st.selectbox(f"{feat} (Hispanic Origin)", options=[0, 1])
    else:
        input_data[feat] = st.number_input(f"{feat}", step=1.0)

# Convert to H2OFrame
if st.button("Predict Displacement Risk"):
    input_df = pd.DataFrame([input_data])
    h2o_df = h2o.H2OFrame(input_df)

    prediction = model.predict(h2o_df).as_data_frame()
    risk = prediction["predict"][0]
    prob = prediction["p1"][0]

    st.subheader(f"Prediction: {'Displaced' if risk == 1 else 'Not Displaced'}")
    st.write(f"Probability of displacement: **{prob:.2%}**")

    # Optional explanation
    if st.checkbox("Show Feature Importance (Optional)"):
        try:
            imp = model.varimp(use_pandas=True)
            st.subheader("Top Important Features")
            st.dataframe(imp[['variable', 'percentage']].head(10))
        except:
            st.warning("Feature importance is not available for this model.")

