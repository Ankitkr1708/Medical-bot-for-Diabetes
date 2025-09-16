# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests
import json

app = Flask(__name__)
CORS(app)

try:
    diabetes_model = joblib.load('diabetes_model.pkl')
    print("Diabetes prediction model loaded successfully.")
except FileNotFoundError:
    print("Error: 'diabetes_model.pkl' not found. Please run train_model.py first.")
    diabetes_model = None

def extract_features_from_text(text):
    features = {
        'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 25, 'Smoker': 0,
        'Stroke': 0, 'HeartDiseaseorAttack': 0, 'PhysActivity': 1, 'Fruits': 1,
        'Veggies': 1, 'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
        'GenHlth': 2, 'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0, 'Sex': 1,
        'Age': 7, 'Education': 6, 'Income': 8
    }

    lower_text = text.lower()

    if 'high bp' in lower_text or 'high blood pressure' in lower_text:
        features['HighBP'] = 1
    if 'high cholesterol' in lower_text:
        features['HighChol'] = 1
    if 'smoker' in lower_text or 'smoke' in lower_text:
        features['Smoker'] = 1
    if 'stroke' in lower_text:
        features['Stroke'] = 1
    if 'heart attack' in lower_text or 'heart disease' in lower_text:
        features['HeartDiseaseorAttack'] = 1
    if 'no exercise' in lower_text or 'sedentary' in lower_text:
        features['PhysActivity'] = 0

    try:
        bmi_search = [word for word in lower_text.split() if word.isdigit()]
        if 'bmi' in lower_text and bmi_search:
            features['BMI'] = int(bmi_search[0])
    except:
        pass

    print(f"Extracted Features: {features}")
    return np.array(list(features.values())).reshape(1, -1)

def query_ollama(prompt):
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llama3:latest",
                "prompt": prompt,
                "stream": False
            },
        )
        response.raise_for_status()

        full_response = response.json().get('response', '')

        json_start = full_response.find('{')
        json_end = full_response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = full_response[json_start:json_end]
            return json.loads(json_str)
        else:
            return {"error": "Failed to parse LLM response.", "raw_response": full_response}

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return {"error": "Could not connect to Ollama server at http://localhost:11434. Is it running?"}
    except json.JSONDecodeError:
        print("Failed to decode JSON from Ollama response.")
        return {"error": "Invalid JSON format from LLM.", "raw_response": full_response}


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    user_input = data.get('message')

    if not user_input or not diabetes_model:
        return jsonify({"error": "Invalid input or model not loaded"}), 400

    # 1. Get statistical risk from the trained ML model
    features = extract_features_from_text(user_input)
    prediction_proba = diabetes_model.predict_proba(features)
    diabetes_risk_percentage = round(prediction_proba[0][1] * 100)

    # 2. Create a detailed prompt for the Ollama Llama model
    prompt = f"""
    You are a helpful, empathetic AI medical assistant.
    A user has provided the following health information: "{user_input}"
    Based on a machine learning model, their statistical risk of having or developing diabetes is {diabetes_risk_percentage}%.

    Your task is to respond in a structured JSON format. Do not add any text before or after the JSON object.
    The JSON object must have a single key "diseases" which is an array.
    Inside the array, create one JSON object with the following keys:
    - "name": A string, set to "Diabetes Risk Assessment".
    - "chance": A number, set to {diabetes_risk_percentage}.
    - "suggestions": An array of strings. Provide 3-4 clear, actionable lifestyle suggestions based on the user's input and their risk level. Be encouraging and supportive.
    - "medicines": An array of strings. List 2-3 common medications or treatments related to diabetes management. **Crucially, add the phrase "(Consult a Doctor)" after each medication name.**

    Example of a valid response format:
    {{
      "diseases": [
        {{
          "name": "Diabetes Risk Assessment",
          "chance": {diabetes_risk_percentage},
          "suggestions": ["Monitor blood sugar levels regularly.", "Incorporate at least 30 minutes of moderate exercise daily.", "Focus on a balanced diet rich in fiber and low in processed sugars."],
          "medicines": ["Metformin (Consult a Doctor)", "Insulin therapy (Consult a Doctor)"]
        }}
      ]
    }}

    Now, generate the JSON response based on the user's input and their {diabetes_risk_percentage}% risk.
    """

    # 3. Query Ollama and get the structured response
    ollama_response = query_ollama(prompt)

    if "error" in ollama_response:
        return jsonify(ollama_response), 500

    return jsonify(ollama_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
