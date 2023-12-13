# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1

# Copy the requirements.txt and .env files into the container at /app
COPY requirements.txt ./
COPY .env ./

# Copy the credentials file into the container at /app
COPY credentials.json /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables from .env file
ENV $(cat .env | grep -v ^# | xargs)

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]
