import numpy as np
from flask import Flask, request, render_template
from joblib import load

# Create Flask app
flask_app = Flask(__name__)
scaler = load("scaler.pkl")  # Load the scaler
model = load("heart_disease_model.pkl")  # Load the trained model

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract the expected 13 input features (adjust the feature names as needed)
        float_features = [
            float(request.form.get("male")),  # 1st feature
            float(request.form.get("age")),  # 2nd feature
            float(request.form.get("education")),  # 3rd feature
            float(request.form.get("currentSmoker")),  # 4th feature
            float(request.form.get("cigsPerDay")),  # 5th feature
            float(request.form.get("BPMeds")),  # 6th feature
            float(request.form.get("prevalentStroke")),  # 7th feature
            float(request.form.get("prevalentHyp")),  # 8th feature
            float(request.form.get("diabetes")),  # 9th feature
            float(request.form.get("totChol")),  # 10th feature
            float(request.form.get("sysBP")),  # 11th feature
            float(request.form.get("diaBP")),  # 12th feature
            float(request.form.get("BMI"))  # 13th feature
        ]
        
        # Convert features into a 2D array (matching the scaler's expected input)
        features = np.array(float_features).reshape(1, -1)
        
        # Apply scaling
        scaled_data = scaler.transform(features)
        
        # Make the prediction using the trained model
        prediction = model.predict(scaled_data)[0]

        # Output the result
        result = "CHD Positive" if prediction == 1 else "CHD Negative"
        
        return render_template("index.html", prediction_text=f"The predicted outcome is: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    flask_app.run(debug=True)
