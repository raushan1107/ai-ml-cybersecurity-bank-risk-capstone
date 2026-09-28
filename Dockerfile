# CAMEL Sentinel — production image (FastAPI backend + Streamlit UI + trained models).
# Design and runbook: docs/06_deployment_containerization.md
#
#   docker build -t camel-sentinel:v2.0 .
#   docker run -d -p 8501:8501 -p 8000:8000 --env-file .env camel-sentinel:v2.0
#
# Secrets are NEVER copied into the image (.env is in .dockerignore); pass them at run time.

FROM python:3.10-slim

# Runtime behaviour: no .pyc files, unbuffered logs (so `docker logs` is live),
# no pip cache, and Streamlit without telemetry or the first-run email prompt.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    MPLCONFIGDIR=/tmp/matplotlib \
    MODE=all \
    CAMEL_BACKEND_URL=http://localhost:8000

# System libraries: libgomp1 (LightGBM / scikit-learn OpenMP), libasound2 + libssl3
# (Azure Speech SDK). No curl: the health check uses Python (fewer CVEs, docs/09).
# Clean apt lists in the same layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libasound2 libssl3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies first — this layer is cached and only rebuilt when requirements change.
#    CPU-only torch avoids ~2.5 GB of CUDA libraries we never use. "+cpu" pins the CPU
#    build; PyPI stays available as an extra index for everything else.
COPY requirements-docker.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
 && pip install torch-geometric==2.8.0.post1 \
 && pip install -r requirements-docker.txt \
 && pip uninstall -y pip
# ↑ pip isn't needed at run time. Removing it drops the msgpack/setuptools copies vendored
#   inside pip (2 HIGH CVEs in the Trivy scan) and stops anyone who gets a shell from
#   installing tools. (Same layer as the installs, so nothing is left behind.)

# 2) Least privilege: an unprivileged user, created BEFORE copying code so files can be
#    copied already owned by it (COPY --chown). A later `chown -R` would duplicate every
#    file into a new layer (it cost 23.7 MB in v1.0). sed strips Windows line endings.
COPY docker/docker-entrypoint.sh docker/healthcheck.sh /usr/local/bin/
RUN useradd --create-home --uid 10001 camel \
 && sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh /usr/local/bin/healthcheck.sh \
 && chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/healthcheck.sh

# 3) Models and data change rarely; application code changes most — so it goes last.
COPY --chown=camel:camel models/ models/
COPY --chown=camel:camel data/ data/
COPY --chown=camel:camel notebooks/ notebooks/
COPY --chown=camel:camel docs/ docs/
COPY --chown=camel:camel deploy/ deploy/
COPY --chown=camel:camel Dockerfile docker-compose.yml requirements-docker.txt ./
COPY --chown=camel:camel .streamlit/ .streamlit/
COPY --chown=camel:camel src/ src/
COPY --chown=camel:camel backend/ backend/
COPY --chown=camel:camel frontend/ frontend/
USER camel

EXPOSE 8000 8501

# Healthy = the service this MODE owns really works: for all/api the API must report
# model_loaded=true. (An earlier "API || UI" check went healthy before the model loaded.)
# start-period covers model + SHAP/LIME warm-up.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
  CMD /usr/local/bin/healthcheck.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
