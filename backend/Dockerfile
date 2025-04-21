# FROM and WORKDIR remain the same
FROM python:3.10-slim
WORKDIR /app

# Copy backend app
COPY backend/ /app/

# Copy model and encoder files from parent
COPY model_weights.pkl /app/
COPY encoders.pkl /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001
CMD ["python", "app.py"]
