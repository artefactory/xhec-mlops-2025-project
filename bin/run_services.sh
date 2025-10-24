#!/bin/bash

# TODO: Use this file in your Dockerfile to run the services

# prefect server start --host 0.0.0.0 --port 4201 &
# uvicorn src.web_service.main:app --reload --port 8080

docker build -t mlops-api -f Dockerfile.app .
docker run -d -p 0.0.0.0:8000:8080 mlops-api

## the commands to run, to add to the readme (explain that the first is to build the image and the second to run it)
