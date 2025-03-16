import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the saved model artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('model_artifacts/cervical_cancer_best_model.joblib')
    scaler = joblib.load('model_artifacts/cervical_cancer_scaler.joblib')
    feature_names = joblib.load('model_artifacts/feature_names.joblib')
    return model, scaler, feature_names

# Load model and artifacts
model, scaler, feature_names = load_model_artifacts()

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Streamlit app title
st.title("Cervical Cancer Risk Assessment Tool")
st.write("""
This tool helps healthcare providers assess a patient's risk of cervical cancer based on medical history and behavioral factors.
It provides a clear risk prediction and explains the factors contributing to the prediction.
""")

# Sidebar for user input
st.sidebar.header("Patient Information")

# Toggle for custom input
use_custom_input = st.sidebar.checkbox("Use Custom Input (Type Values Manually)")

# Function to dynamically generate input fields based on feature names
def get_user_input(feature_names, use_custom_input):
    input_data = {}
    for feature in feature_names:
        if use_custom_input:
            # Allow manual input for custom values
            input_data[feature] = st.sidebar.number_input(feature, value=0.0)
        else:
            # Use sliders/dropdowns for predefined ranges
            if feature == "Age":
                input_data[feature] = st.sidebar.slider(feature, 13, 84, 30)
            elif "years" in feature.lower():
                input_data[feature] = st.sidebar.slider(feature, 0.0, 50.0, 0.0, step=0.1)
            elif "number" in feature.lower():
                input_data[feature] = st.sidebar.slider(feature, 0, 4, 0)
            else:
                # Convert "No" to 0 and "Yes" to 1
                input_data[feature] = 1 if st.sidebar.selectbox(feature, ["No", "Yes"]) == "Yes" else 0
    return pd.DataFrame([input_data])

# Get user input
user_input = get_user_input(feature_names, use_custom_input)

# Display user input
st.subheader("Patient Input Summary")
st.write(user_input)

# Preprocess input data
def preprocess_input(input_df, feature_names):
    # Ensure the input DataFrame has all features in the correct order
    input_df = input_df[feature_names]
    
    # Convert all columns to numeric (just in case)
    input_df = input_df.apply(pd.to_numeric)
    
    # Scale the features
    scaled_input = scaler.transform(input_df)
    return scaled_input

# Preprocess user input
scaled_input = preprocess_input(user_input, feature_names)

# Make predictions
def predict_risk(model, input_data):
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction_class = 1 if prediction_proba >= 0.5 else 0
    return prediction_class, prediction_proba

# Predict risk
prediction_class, prediction_proba = predict_risk(model, scaled_input)

# Display prediction with visual feedback
st.subheader("Risk Prediction")
risk_level = "High Risk" if prediction_class == 1 else "Low Risk"
risk_color = "red" if risk_level == "High Risk" else "green"

st.markdown(f"""
<div style="background-color:{risk_color}; padding:10px; border-radius:5px;">
    <h3 style="color:white; text-align:center;">{risk_level}</h3>
</div>
""", unsafe_allow_html=True)

st.write(f"**Probability of Cervical Cancer:** {prediction_proba:.2%}")

# Provide actionable recommendations
st.subheader("Recommendations")
if risk_level == "High Risk":
    st.write("""
    - **Immediate Action**: Recommend a Pap smear or HPV test.
    - **Lifestyle Changes**: Encourage smoking cessation and safe sexual practices.
    - **Follow-Up**: Schedule a follow-up appointment within 3 months.
    """)
else:
    st.write("""
    - **Routine Screening**: Continue regular cervical cancer screening as per guidelines.
    - **Preventive Measures**: Encourage HPV vaccination if not already administered.
    """)

# SHAP Explanations
st.subheader("Explanation of the Prediction")

# Generate SHAP values
shap_values = explainer.shap_values(scaled_input)

# SHAP Summary Plot
st.write("**Feature Importance**")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, scaled_input, feature_names=feature_names, plot_type="bar", show=False)
st.pyplot(plt)

# SHAP Force Plot for the current prediction
st.write("**How Each Feature Contributed to This Prediction**")
plt.figure(figsize=(10, 4))
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    scaled_input[0],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
st.pyplot(plt)

# SHAP Dependence Plots for Top 3 Features
st.write("**How Key Features Affect the Prediction**")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[-3:][::-1]
top_features = [feature_names[i] for i in top_indices]

for feature in top_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values,
        scaled_input,
        feature_names=feature_names,
        show=False
    )
    st.pyplot(plt)

# Footer
st.write("---")
st.write("**Note:** This tool is for educational and informational purposes only. Consult a healthcare professional for medical advice.")