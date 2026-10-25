# Python 3.12 supports the supplied package versions and is still supported
FROM python:3.12-slim

WORKDIR /app

# XGBoost needs the OpenMP runtime in the Linux container
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 7860
# Two workers are sufficient for the course-sized deployment
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:superkart_api"]
