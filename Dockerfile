# Use an official Python runtime as a parent image
FROM python:3.10
RUN apt-get update && apt-get install -y libgl1-mesa-glx


# Set the working directory to /app
WORKDIR /app


COPY . /app


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Install the necessary system libraries for OpenCV
RUN pip install google-cloud-vision


# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application using Uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port","8000","--workers","4"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

