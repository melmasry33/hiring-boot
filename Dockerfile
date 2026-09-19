# Slim image on purpose: no Chromium, no Playwright, no build toolchain.
# The cloud path talks to LinkedIn over plain HTTP (see app/jobs.py), which
# keeps the image around 200 MB instead of 1.5 GB and the cold start in seconds.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Africa/Cairo

WORKDIR /srv

# CA certificates are needed for TLS to Telegram, the LLM provider and SMTP.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tzdata gosu fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

# Requirements first so dependency layers cache across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/
COPY tests/ ./tests/

# Run unprivileged. /data is where a Railway Volume gets mounted.
RUN useradd --create-home --uid 10001 agent \
 && mkdir -p /data /tmp/generated_cvs \
 && chown -R agent:agent /srv /data /tmp/generated_cvs

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

ENV PYTHONPATH=/srv/app \
    DATA_DIR=/data \
    GENERATED_CVS_DIR=/tmp/generated_cvs \
    PORT=8080

EXPOSE 8080

CMD ["python", "-u", "app/bot.py"]
