#!/bin/sh
# Creates .venv in this folder and installs realtime-translation in it (editable).
# On Apple Silicon the MLX backends (Voxtral MLX, mlx-whisper) are installed too ;
# on Linux with an NVIDIA GPU, the GPU server backends (Voxtral through transformers).
set -e
cd "$(dirname "$0")"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

EXTRAS=""
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    EXTRAS="[mlx]"
elif [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    EXTRAS="[gpu]"
fi

if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON_VERSION" .venv
    uv pip install --python .venv/bin/python -e ".$EXTRAS"
else
    PY="$(command -v "python$PYTHON_VERSION" || command -v python3)"
    "$PY" -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e ".$EXTRAS"
fi

echo
echo "Installed. Models are downloaded on first run (Voxtral ~3 GB, NLLB 600M ~2.5 GB)."
echo "  .venv/bin/rt-translate --list-devices"
echo "  .venv/bin/rt-translate --device \"<device name>\" --lan fr --target-language en"
