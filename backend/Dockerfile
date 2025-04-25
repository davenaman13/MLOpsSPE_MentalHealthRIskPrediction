FROM python:3.10-slim

WORKDIR /app

# First copy ONLY requirements.txt to cache dependencies
COPY requirements.txt .

# Install dependencies including test packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt pytest pytest-cov

# Then copy the rest of the application
COPY . .

EXPOSE 5001

CMD ["python", "app.py"]