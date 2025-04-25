from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

from utils import load_encoders, preprocess_input

app = Flask(__name__)

# Set base directory to the current file's location
base_dir = os.path.dirname(__file__)

# Load model and encoders with absolute paths
with open(os.path.join(base_dir, 'model_weights.pkl'), 'rb') as f:
    model = pickle.load(f)

encoders = load_encoders(path=os.path.join(base_dir, 'encoders.pkl'))

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

# @app.route('/health')
# def health_check():
#     return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
