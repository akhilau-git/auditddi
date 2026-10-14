FROM python:3.10-slim
ARG PYTORCH_VERSION=2.5.1
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# The project uses pre-built wheels.  libgomp is required by numerical wheels;
# the small X11 runtime set is required by RDKit's drawing module used by
# explainability endpoints and test rendering.
# A full compiler toolchain is unnecessary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install first (Docker layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==${PYTORCH_VERSION}" \
    && pip install --no-cache-dir -r requirements.txt

# Copy BOTH backend code AND src/ into the image — this is the actual fix
COPY backend/ ./backend/
COPY src/ ./src/

WORKDIR /app/backend
RUN useradd --create-home --shell /usr/sbin/nologin auditddi \
    && mkdir -p /app/state \
    && chown auditddi:auditddi /app/state
USER auditddi

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/ready', timeout=2)"

# The default is the Compose port, while hosted services can inject PORT.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
