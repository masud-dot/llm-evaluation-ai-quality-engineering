# Optional OSS trace backend for Project 5.
FROM python:3.12-slim
WORKDIR /app
COPY requirements/lock-adapters.txt .
RUN pip install --no-cache-dir -r lock-adapters.txt
ENV PHOENIX_WORKING_DIR=/data PHOENIX_PORT=6006
EXPOSE 6006
CMD ["phoenix", "serve"]
