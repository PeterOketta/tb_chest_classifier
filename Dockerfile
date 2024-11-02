# Use an official Python image as the base
FROM python:3.9-slim

# Set environment variables to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install curl and other required system dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model file using the pre-authenticated URL
RUN mkdir -p model && \
    curl -L -o model/classification_model.keras "https://storage.cloud.google.com/classification_model_tb/classification_model.keras"

# Copy the entire project directory into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Define the command to run the application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]