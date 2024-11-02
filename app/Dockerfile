# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --no-cache-dir -r /app/app/requirements.txt

# Install wget to download the model
RUN apt-get update && apt-get install -y wget

# Download the model from GitHub Releases
RUN wget -O /app/model/classification_model.keras \
    "https://github.com/Marconi-Lab/TB-Screening/releases/download/classification_v2/1_classification_my_model.keras"

# Copy the application code
COPY app /app/app

# Set up the entry point to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
