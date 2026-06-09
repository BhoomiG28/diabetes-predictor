from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        int(request.form["pregnancies"]),
        int(request.form["glucose"]),
        int(request.form["bloodpressure"]),
        int(request.form["skinthickness"]),
        int(request.form["insulin"]),
        float(request.form["bmi"]),
        float(request.form["dpf"]),
        int(request.form["age"])
    ]

    data = scaler.transform([data])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "⚠️ Diabetic"
    else:
        result = "✅ Not Diabetic"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)