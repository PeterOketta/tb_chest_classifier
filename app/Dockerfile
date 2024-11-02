# Use an official Python image as a base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/classification_model.keras

# Install required packages and tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install them
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source code
COPY app/ app/

# Create the model directory
RUN mkdir -p /app/model

# Download the model from GCP bucket
RUN wget -O $MODEL_PATH "https://storage.googleapis.com/classification_model_tb/classification_model.keras"

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the default command to run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
