# backend/app.py

from flask import Flask, request, jsonify
import pandas as pd
import torch
from model import MentalHealthNN
from utils import load_encoders, preprocess_input

app = Flask(__name__)

# Load encoders
encoders = load_encoders()

# Initialize and load model
dummy_input = torch.randn(1, len(encoders))
model = MentalHealthNN(input_dim=dummy_input.shape[1], hidden_dim=32, output_dim=1)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.json
        user_df = pd.DataFrame([user_data])
        print("[BACKEND] DataFrame:\n", user_df)

        user_encoded = preprocess_input(user_df, encoders)
        print("[BACKEND] Encoded input:\n", user_encoded)

        input_tensor = torch.tensor(user_encoded.values, dtype=torch.float32)
        with torch.no_grad():
            pred = model(input_tensor).item()
        result = "Will Seek Treatment" if pred >= 0.5 else "Will Not Seek Treatment"
        print("[BACKEND] Prediction result:", result)
        return jsonify({'prediction': result})
    except Exception as e:
        print("[BACKEND] Error occurred:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001)
