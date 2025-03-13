# ! havent tested this dockerfile yet

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install the local package
RUN pip install -e .

# Expose ports for Gradio and FastAPI
EXPOSE 7860 8234

# Install Ollama (optional, as it requires separate setup)
# RUN curl https://ollama.ai/install.sh | sh

# Set the default command to run the Gradio app
CMD ["python", "-m", "rufus.gradio_app"]