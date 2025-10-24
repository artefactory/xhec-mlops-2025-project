# Make sure to check bin/run_services.sh, which can be used here

# Do not forget to expose the right ports! (Check the PR_4.md)


FROM python:3.10.13-slim
WORKDIR /app_home
COPY ./requirements-dev.in /app_home/requirements-dev.in
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app_home/requirements-dev.in
# Copy the entire repository
COPY . /app_home

EXPOSE 8080

CMD ["uvicorn", "src.web_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
