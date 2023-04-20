# Use the official Python image as the base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the app runs on
EXPOSE 5000

# Install ploomber
RUN pip install ploomber

# Remove files ending in .metadata from the notebooks folder
RUN find notebooks -type f -name "*.metadata" -exec rm -f {} \;

# Create a shell script to run the commands in sequence
RUN echo '#!/bin/sh' > run.sh && \
    echo 'ploomber build' >> run.sh && \
    echo 'python app.py' >> run.sh && \
    chmod +x run.sh

# Execute the script when the container starts
CMD ["./run.sh"]