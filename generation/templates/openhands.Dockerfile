FROM python:3.12-slim
ENV LITELLM_LOCAL_MODEL_COST_MAP=True OPENHANDS_SUPPRESS_BANNER=1
ARG DEBIAN_MIRROR=https://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends libreoffice-calc curl coreutils git ripgrep \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir openpyxl==3.1.5 \
    && python -m venv /opt/openhands-sdk-venv \
    && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir \
       openhands-sdk=={{sdk_version}} openhands-tools=={{sdk_version}} fastapi openpyxl==3.1.5 \
    && /opt/openhands-sdk-venv/bin/python -c 'import openhands.sdk; print(openhands.sdk.__version__)'
# Harbor finds this preinstalled SDK and uploads only its small runner script.
# All shared dependencies precede task-specific inputs so Docker can reuse layers.
WORKDIR /app
COPY input.xlsx /app/input.xlsx
COPY public/ /app/public/
