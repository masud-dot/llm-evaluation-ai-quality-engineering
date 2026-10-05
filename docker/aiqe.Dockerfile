# Online evaluation worker for Project 5.
FROM python:3.12-slim
WORKDIR /app
COPY requirements/lock-core-providers-dev.txt requirements/
RUN pip install --no-cache-dir \
    -r requirements/lock-core-providers-dev.txt \
    opentelemetry-exporter-otlp-proto-http==1.44.0
COPY pyproject.toml ./
COPY src ./src
COPY systems ./systems
COPY projects ./projects
RUN pip install --no-cache-dir --no-deps .
CMD ["python", "projects/p5-production-platform/worker.py", \
     "--input", "/data/conversations.jsonl", \
     "--scores", "/data/scores.jsonl", \
     "--triage", "/data/triage.jsonl"]
