#!/bin/bash

# TODO: Use this file in your Dockerfile to run the services

# prefect server start --host 0.0.0.0 --port 4201 &
# uvicorn src.web_service.main:app --reload --port 8080

docker build -t mlops-api -f Dockerfile.app .
docker run -d -p 0.0.0.0:8000:8080 mlops-api

## the commands to run, to add to the readme (explain that the first is to build the image and the second to run it)
# docker ps to check if container running
# curl http://127.0.0.1:8080 to test the get api inside docker app
# head -n 20 src/modelling/train_flow.py to retrieve the first 20 lines of the train flow file, to check if indeed in docker app
# ls src/ to list all files under src (all relevant apps for us)
