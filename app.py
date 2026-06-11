from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_results = pickle.load(open("model_results.pkl", "rb"))
best_model_name = pickle.load(open("best_model_name.pkl", "rb"))

FEATURE_NAMES = ["Pregnancies", "Glucose", "Blood Pressure",
                 "Skin Thickness", "Insulin", "BMI",
                 "Diabetes Pedigree Function", "Age"]

def get_risk_factors(data):
    factors = []
    if data[1] > 140: factors.append("High Glucose Level")
    if data[5] > 30: factors.append("High BMI")
    if data[7] > 45: factors.append("High Age")
    if data[0] > 5: factors.append("High Number of Pregnancies")
    if data[4] > 150: factors.append("High Insulin Level")
    if data[2] > 80: factors.append("High Blood Pressure")
    return factors if factors else ["No major risk factors detected"]

def validate_inputs(data):
    errors = []
    if data[0] < 0 or data[0] > 20:
        errors.append("Pregnancies must be between 0 and 20")
    if data[1] < 50 or data[1] > 300:
        errors.append("Glucose must be between 50 and 300")
    if data[2] < 30 or data[2] > 150:
        errors.append("Blood Pressure must be between 30 and 150")
    if data[3] < 0 or data[3] > 100:
        errors.append("Skin Thickness must be between 0 and 100")
    if data[4] < 0 or data[4] > 900:
        errors.append("Insulin must be between 0 and 900")
    if data[5] < 10 or data[5] > 70:
        errors.append("BMI must be between 10 and 70")
    if data[6] < 0 or data[6] > 3:
        errors.append("Diabetes Pedigree Function must be between 0 and 3")
    if data[7] < 1 or data[7] > 120:
        errors.append("Age must be between 1 and 120")
    return errors

@app.route("/")
def home():
    return render_template("index.html",
                         model_results=model_results,
                         best_model_name=best_model_name)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skinthickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        # Validate
        errors = validate_inputs(data)
        if errors:
            return render_template("index.html",
                                 errors=errors,
                                 model_results=model_results,
                                 best_model_name=best_model_name)

        # Scale and predict
        scaled = scaler.transform([data])
        prediction = model.predict(scaled)[0]
        probability = round(model.predict_proba(scaled)[0][1] * 100, 1)

        # Risk factors
        risk_factors = get_risk_factors(data)

        # Risk level
        if probability < 30:
            risk_level = "Low"
            risk_color = "green"
        elif probability < 60:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template("index.html",
                             result=result,
                             probability=probability,
                             risk_level=risk_level,
                             risk_color=risk_color,
                             risk_factors=risk_factors,
                             model_results=model_results,
                             best_model_name=best_model_name)

    except ValueError:
        return render_template("index.html",
                             errors=["Please fill all fields with valid numbers"],
                             model_results=model_results,
                             best_model_name=best_model_name)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)