from flask import Flask, render_template, request
import requests

app = Flask(__name__)

BACKEND_URL = "http://mental-health-backend:5001/predict"  # Backend server URL

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            response = requests.post(BACKEND_URL, json=form_data)
            prediction = response.json().get("prediction")
            print("[FRONTEND] Got prediction:", prediction)  # Debug line
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5003, debug=True)
