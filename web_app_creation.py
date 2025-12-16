from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and label encoder (GROUP2)
model = joblib.load("crop_model_GROUP2.pkl")
label_encoder = joblib.load("label_encoder_GROUP2.pkl")


@app.route("/")
def home():
    return """
    <h2>🌱 Crop Recommendation System (GROUP2)</h2>
    <form action="/predict" method="post">
        Nitrogen: <input name="N" type="number" step="any" required><br><br>
        Phosphorus: <input name="P" type="number" step="any" required><br><br>
        Potassium: <input name="K" type="number" step="any" required><br><br>
        Temperature: <input name="temperature" type="number" step="any" required><br><br>
        Humidity: <input name="humidity" type="number" step="any" required><br><br>
        pH: <input name="ph" type="number" step="any" required><br><br>
        Rainfall: <input name="rainfall" type="number" step="any" required><br><br>
        <button type="submit">Predict Crop</button>
    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"]),
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        return f"<h3>✅ Recommended Crop: <b>{crop_name}</b></h3>"

    except Exception as e:
        return f"<h3>❌ Error: {str(e)}</h3>"


# API endpoint (for mobile / global access)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    features = np.array([[
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"],
    ]])

    prediction = model.predict(features)
    crop = label_encoder.inverse_transform(prediction)[0]

    return jsonify({
        "recommended_crop": crop,
        "group": "GROUP2"
    })


if __name__ == "__main__":
    app.run(debug=True)

