FROM python:3.11-slim
ARG DEBIAN_MIRROR=https://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends libreoffice-calc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir openpyxl==3.1.5

# Install once in a shared image layer, before any task-specific files.
# The official platform package includes the native CLI and its helper binaries;
# no Node/npm installation is needed in the final image or at trial startup.
ARG CODEX_PACKAGE_BASE_URL=https://registry.npmjs.org/@openai/codex/-
RUN python - <<'PY'
import base64
import hashlib
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

version = "0.153.4"
packages = {
    "x86_64": ("x64", "x86_64-unknown-linux-musl", "x1EcwBlY3AObM1VTUHNM2AzAJQsyreGdagpF+qFiYi/Oa30VBktvvG0C6tLtCzqW6hjZNWkGZQWmeVk7MuJKWg=="),
    "aarch64": ("arm64", "aarch64-unknown-linux-musl", "QKdjYLYV4hXIuUQDP3P6F4NXuWFoKo9WUoV4nAREIx55kiUyi8UsYdsVobkeXir5n/maEQgYMCKLHVma4rNPiw=="),
}
arch, target, expected = packages[platform.machine()]
url = f"{os.environ['CODEX_PACKAGE_BASE_URL'].rstrip('/')}/codex-{version}-linux-{arch}.tgz"
with tempfile.TemporaryDirectory() as directory:
    archive = Path(directory) / "codex.tgz"
    with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    with archive.open("rb") as stream:
        actual = base64.b64encode(hashlib.file_digest(stream, "sha512").digest()).decode()
    if actual != expected:
        raise ValueError("Codex package checksum mismatch")
    with tarfile.open(archive) as package:
        package.extractall(directory, filter="data")
    shutil.move(str(Path(directory) / "package/vendor" / target), "/opt/codex")
PY
ENV PATH="/opt/codex/bin:/opt/codex/codex-path:${PATH}"
RUN test "$(codex --version)" = "codex-cli 0.153.4" && rg --version
WORKDIR /app
COPY input.xlsx /app/input.xlsx
COPY public/ /app/public/
