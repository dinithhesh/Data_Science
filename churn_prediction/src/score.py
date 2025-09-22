import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string


app = Flask(__name__)

# Load your trained model
model = joblib.load("Best Churn Model.pkl")

html_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Model Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Roboto', sans-serif;
        }
        body {
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: #fff;
        }
        .container {
            background: rgba(0, 0, 0, 0.7);
            padding: 40px 50px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 400px;
            text-align: center;
            animation: fadeIn 1s ease-in-out;
        }
        h1 {
            margin-bottom: 30px;
            font-size: 2em;
            color: #ffeb3b;
            text-shadow: 1px 1px 5px rgba(0,0,0,0.3);
        }
        label {
            display: block;
            margin: 15px 0 5px;
            font-weight: bold;
            font-size: 1.1em;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px 15px;
            border-radius: 8px;
            border: none;
            outline: none;
            font-size: 1em;
            margin-bottom: 15px;
            transition: 0.3s;
        }
        input[type="number"]:focus {
            box-shadow: 0 0 10px #ffeb3b;
        }
        button {
            padding: 12px 25px;
            font-size: 1.1em;
            background: #ffeb3b;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
            font-weight: bold;
            color: #000;
        }
        button:hover {
            background: #fdd835;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        #result {
            margin-top: 25px;
            font-size: 1.3em;
            font-weight: bold;
            color: #00e676;
            min-height: 1.5em;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Your Model</h1>
        <label for="recency">Recency:</label>
        <input type="number" id="recency" value="30">

        <label for="frequency">Frequency:</label>
        <input type="number" id="frequency" value="5">

        <label for="monetary">Monetary:</label>
        <input type="number" id="monetary" value="1200">

        <button onclick="sendPrediction()">Predict</button>
        <h2 id="result"></h2>
    </div>

    <script>
        async function sendPrediction() {
            const recency = Number(document.getElementById("recency").value);
            const frequency = Number(document.getElementById("frequency").value);
            const monetary = Number(document.getElementById("monetary").value);

            const data = { features: [recency, frequency, monetary] };

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                document.getElementById("result").innerText = "Prediction: " + result.prediction;
            } catch (error) {
                console.error("Error:", error);
                document.getElementById("result").innerText = "Error connecting to API";
            }
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(html_page)


# Predict route for POST requests
@app.route("/predict", methods=["POST"])

def predict():
    data = request.json  # Expecting JSON input
    features = np.array(data["features"]).reshape(1, -1)  # e.g. [[recency, frequency, monetary]]
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

# This block must start at the leftmost column
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


