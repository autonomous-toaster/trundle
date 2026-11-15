# trundle in a distroless image: no shell, no package manager, nonroot.
#
# The runtime image must match the interpreter the wheels were built for, so the
# builder and the base are both Debian 12 / Python 3.11. onnxruntime needs
# libgomp and libstdc++, which the distroless python3 image already carries.
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --prefix=/install --no-warn-script-location -c /tmp/constraints.txt .


FROM gcr.io/distroless/python3-debian12:nonroot AS runtime

ARG PYTHON_VERSION=3.11

COPY --from=build /install/lib/python${PYTHON_VERSION}/site-packages /app/site-packages
# Model downloads are plain HTTPS; do not rely on the base image having roots.
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
# Owned by nonroot so a named volume mounted here is writable on first start.
COPY --chown=nonroot:nonroot docker/models.keep /models/.keep

ENV PYTHONPATH=/app/site-packages \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TRUNDLE_HOST=0.0.0.0 \
    TRUNDLE_PORT=8000 \
    TRUNDLE_MODEL_DIR=/models

VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python3", "-c", "import os,urllib.request,sys; port=os.environ.get('TRUNDLE_PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["python3", "-m", "trundle"]
CMD ["serve"]
