FROM python:3.9

WORKDIR /app

# First copy ONLY requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --retries 5 --timeout 60 -r requirements.txt pytest pytest-cov

# Then copy the rest
COPY . .

EXPOSE 5003

CMD ["python", "app.py"]