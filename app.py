from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model

model = joblib.load("model/disease_model.pkl")
gender_encoder = joblib.load("model/gender_encoder.pkl")
disease_encoder = joblib.load("model/disease_encoder.pkl")

# Disease Information

disease_info = {

    "Flu": {
        "description": "Influenza is a viral infection affecting the respiratory system.",
        "medicine": "Paracetamol, Rest, Plenty of Fluids",
        "precaution": "Take proper rest, drink water and consult a doctor."
    },

    "Diabetes": {
        "description": "High blood sugar level due to insulin problems.",
        "medicine": "Metformin (Doctor Advice Required)",
        "precaution": "Avoid sugar and exercise regularly."
    },

    "Hypertension": {
        "description": "High blood pressure.",
        "medicine": "Blood Pressure Medicines",
        "precaution": "Reduce salt intake and exercise."
    }

}

# Home

@app.route("/")
def home():
    return render_template("index.html")


# Prediction

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "abdominal_pain":0,
        "anxiety":0,
        "back_pain":0,
        "bloating":0,
        "blurred_vision":0,
        "body_pain":0,
        "burning_urination":0,
        "chest_pain":0,
        "chills":0,
        "cold_hands":0,
        "confusion":0,
        "constipation":0,
        "cough":0,
        "dark_urine":0,
        "diarrhea":0,
        "dizziness":0,
        "dry_mouth":0,
        "ear_pain":0,
        "eye_redness":0,
        "fainting":0,
        "fatigue":0,
        "fever":0,
        "frequent_urination":0,
        "headache":0,
        "heartburn":0,
        "insomnia":0,
        "itching":0,
        "joint_pain":0,
        "loss_of_appetite":0,
        "loss_of_smell":0,
        "muscle_pain":0,
        "nausea":0,
        "neck_pain":0,
        "palpitations":0,
        "rash":0,
        "runny_nose":0,
        "sensitivity_to_light":0,
        "shortness_of_breath":0,
        "skin_peeling":0,
        "sore_throat":0,
        "stomach_cramps":0,
        "sweating":0,
        "swollen_glands":0,
        "vomiting":0,
        "weight_gain":0,
        "weight_loss":0,
        "wheezing":0,
        "yellow_skin":0,
        "Age":0,
        "Gender":0
    }

    # Age
    data["Age"] = int(request.form.get("Age"))

    # Gender
    gender = request.form.get("Gender")

    try:
        data["Gender"] = gender_encoder.transform([gender])[0]
    except:
        data["Gender"] = 1 if gender == "Female" else 0

    # Symptoms
    symptoms = request.form.getlist("symptoms")

    for symptom in symptoms:
        if symptom in data:
            data[symptom] = 1

    df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(df)[0]

    try:
        disease = disease_encoder.inverse_transform([prediction])[0]
    except:
        disease = str(prediction)

    # Confidence
    try:
        confidence = round(max(model.predict_proba(df)[0]) * 100, 2)
    except:
        confidence = 100

    # Disease Information
    info = disease_info.get(
        disease,
        {
            "description": "Disease information not available.",
            "medicine": "Consult a Doctor.",
            "precaution": "Please consult a healthcare professional for proper diagnosis."
        }
    )

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence,
        description=info["description"],
        medicine=info["medicine"],
        precaution=info["precaution"]
    )


if __name__ == "__main__":
    app.run(debug=True)