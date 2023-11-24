# Use the official Python image as the base image
FROM python:3.10.4-slim

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates

# Install system-level dependencies
RUN apt-get update \
    && apt-get install -y poppler-utils openjdk-11-jre \
    

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

ENTRYPOINT []
CMD ["/bin/bash"]
