from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

default_values_positive = {
    "Age": 35.0,  # Middle-aged individuals are often at higher risk
    "Number of sexual partners": 10.0,  # Higher number of sexual partners increases risk
    "First sexual intercourse": 16.0,  # Younger age at first intercourse increases risk
    "Num of pregnancies": 3.0,  # Higher number of pregnancies may increase risk
    "Smokes": 1.0,  # Smoking increases risk
    "Smokes (years)": 10.0,  # Long-term smoking increases risk
    "Smokes (packs/year)": 5.0,  # Heavy smoking increases risk
    "Hormonal Contraceptives": 1.0,  # Long-term use of hormonal contraceptives increases risk
    "Hormonal Contraceptives (years)": 8.0,  # Longer duration increases risk
    "IUD": 0.0,  # IUD use may not significantly increase risk
    "IUD (years)": 0.0,
    "STDs": 1.0,  # Presence of STDs increases risk
    "STDs (number)": 2.0,  # Multiple STDs increase risk
    "STDs:condylomatosis": 1.0,  # Specific STDs like condylomatosis increase risk
    "STDs:cervical condylomatosis": 1.0,
    "STDs:vaginal condylomatosis": 1.0,
    "STDs:vulvo-perineal condylomatosis": 1.0,
    "STDs:syphilis": 0.0,  # Syphilis may not be directly linked
    "STDs:pelvic inflammatory disease": 1.0,  # Pelvic inflammatory disease increases risk
    "STDs:genital herpes": 1.0,  # Genital herpes increases risk
    "STDs:molluscum contagiosum": 0.0,  # Less common, may not significantly increase risk
    "STDs:AIDS": 0.0,  # AIDS may not be directly linked
    "STDs:HIV": 1.0,  # HIV increases risk
    "STDs:Hepatitis B": 0.0,  # Hepatitis B may not be directly linked
    "STDs:HPV": 1.0,  # HPV is a major risk factor for cervical cancer
    "STDs: Number of diagnosis": 2.0,  # Multiple STD diagnoses increase risk
    "Dx:Cancer": 1.0,  # Previous cancer diagnosis increases risk
    "Dx:CIN": 1.0,  # Cervical intraepithelial neoplasia increases risk
    "Dx:HPV": 1.0,  # HPV diagnosis increases risk
    "Dx": 1.0,  # General diagnosis increases risk
    "Hinselmann": 1.0,  # Positive Hinselmann test indicates risk
    "Schiller": 1.0,  # Positive Schiller test indicates risk
    "Citology": 1.0  # Abnormal cytology indicates risk
}

# Default values for the input fields
default_values_negative = {
    "Age": 18,
    "Number of sexual partners": 4.0,
    "First sexual intercourse": 15.0,
    "Num of pregnancies": 1.0,
    "Smokes": 0.0,
    "Smokes (years)": 0.0,
    "Smokes (packs/year)": 0.0,
    "Hormonal Contraceptives": 0.0,
    "Hormonal Contraceptives (years)": 0.0,
    "IUD": 0.0,
    "IUD (years)": 0.0,
    "STDs": 0.0,
    "STDs (number)": 0.0,
    "STDs:condylomatosis": 0.0,
    "STDs:cervical condylomatosis": 0.0,
    "STDs:vaginal condylomatosis": 0.0,
    "STDs:vulvo-perineal condylomatosis": 0.0,
    "STDs:syphilis": 0.0,
    "STDs:pelvic inflammatory disease": 0.0,
    "STDs:genital herpes": 0.0,
    "STDs:molluscum contagiosum": 0.0,
    "STDs:AIDS": 0.0,
    "STDs:HIV": 0.0,
    "STDs:Hepatitis B": 0.0,
    "STDs:HPV": 0.0,
    "STDs: Number of diagnosis": 0.0,
    "Dx:Cancer": 0.0,
    "Dx:CIN": 0.0,
    "Dx:HPV": 0.0,
    "Dx": 0.0,
    "Hinselmann": 0.0,
    "Schiller": 0.0,
    "Citology": 0.0
}

default_values = default_values_positive

# Load the model and artifacts
model = joblib.load('model_artifacts/cervical_cancer_best_model_gradient_boosting.joblib')
scaler = joblib.load('model_artifacts/cervical_cancer_scaler.joblib')
feature_names = joblib.load('model_artifacts/feature_names.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Initialize prediction as None

    if request.method == 'POST':
        # Get the input data from the form
        input_data = {}
        for feature in feature_names:
            input_data[feature] = float(request.form[feature])

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        try:
            prediction_prob = model.predict_proba(input_scaled)[:, 1]
            prediction = prediction_prob[0]
        except:
            prediction = model.predict(input_scaled)[0]

    # Render the template with prediction, feature_names, and default_values
    return render_template(
        'index.html',
        prediction=prediction,
        feature_names=feature_names,
        default_values=default_values
    )

if __name__ == '__main__':
    app.run(debug=True)