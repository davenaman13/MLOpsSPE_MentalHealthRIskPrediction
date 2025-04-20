# backend/app.py

from flask import Flask, request, jsonify
import pickle
import pandas as pd
from utils import load_encoders, preprocess_input

app = Flask(__name__)

# Load model and encoders
with open('model_weights.pkl', 'rb') as f:
    model = pickle.load(f)

encoders = load_encoders()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.json
        print("[BACKEND] Received data:", user_data)  # Debug
        user_df = pd.DataFrame([user_data])
        print("[BACKEND] DataFrame:\n", user_df)  # Debug

        user_encoded = preprocess_input(user_df, encoders)
        print("[BACKEND] Encoded input:\n", user_encoded)  # Debug

        prediction = model.predict(user_encoded)[0]
        result = "Will Seek Treatment" if prediction == 1 else "Will Not Seek Treatment"
        print("[BACKEND] Prediction result:", result)  # Debug
        return jsonify({'prediction': result})
    except Exception as e:
        print("[BACKEND] Error occurred:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=5001)