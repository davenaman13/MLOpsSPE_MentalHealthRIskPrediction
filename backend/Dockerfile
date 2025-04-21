FROM python:3.10-slim

WORKDIR /app

# Copy backend code into the container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose backend port
EXPOSE 5001

CMD ["python", "app.py"]
