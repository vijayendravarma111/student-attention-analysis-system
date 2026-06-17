# Use a lightweight python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system libraries required by OpenCV (libGL and Glib)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Configure default port for containerized cloud deployment (e.g. Hugging Face Spaces default port)
ENV PORT=7860
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
