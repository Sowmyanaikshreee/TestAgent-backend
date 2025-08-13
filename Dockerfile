# Use official Python base image
FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Install system dependencies (only if needed, here libgl1 is the replacement)
# If you don't need OpenGL at all, you can comment this out
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
