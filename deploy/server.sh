#!/bin/sh
# Installs (first run) and starts the transcription/translation server on a Linux NVIDIA GPU machine.
#
#   WLK_API_TOKEN=<secret> ./deploy/server.sh
#
# Settings (environment variables) :
#   WLK_API_TOKEN    required : clients must send it (rt-translate --token ...)
#   BACKEND          voxtral (default, Voxtral Mini 4B realtime) | faster-whisper | ...
#   MODEL            whisper size for whisper backends (e.g. large-v3-turbo)
#   LAN              source language, default auto
#   TARGET_LANGUAGE  default translation target (clients can override it per session), default en ;
#                    empty = no translation
#   NLLB_SIZE        600M | 1.3B (default)
#   HOST / PORT      default 127.0.0.1 / 8000 (reach it with an SSH tunnel, see deploy/README.md)
#   HF_HOME          model cache, default ~/.cache/huggingface
#   EXTRA_ARGS       extra arguments for the server
set -e
cd "$(dirname "$0")/.."

if [ -z "$WLK_API_TOKEN" ] && [ -z "$ALLOW_NO_TOKEN" ]; then
    echo "WLK_API_TOKEN is not set. Generate one with :  export WLK_API_TOKEN=\$(openssl rand -hex 24)" >&2
    echo "(or set ALLOW_NO_TOKEN=1 if the port is not reachable from the internet)" >&2
    exit 1
fi

if [ ! -x .venv/bin/rt-translate-server ]; then
    command -v ffmpeg >/dev/null 2>&1 || { (apt-get update && apt-get install -y ffmpeg) >/dev/null 2>&1 || \
        echo "warning : ffmpeg not installed (only needed for non-PCM clients)"; }
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -e . "whisperlivekit[translation,voxtral-hf]>=0.2.26"
fi

.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; \
print('GPU :', torch.cuda.get_device_name(0), '%.0f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))"

TARGET_LANGUAGE="${TARGET_LANGUAGE-en}"
set -- --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --backend "${BACKEND:-voxtral}" \
       --pcm-input --lan "${LAN:-auto}"
[ -n "$MODEL" ] && set -- "$@" --model "$MODEL"
[ -n "$TARGET_LANGUAGE" ] && set -- "$@" --target-language "$TARGET_LANGUAGE" --nllb-size "${NLLB_SIZE:-1.3B}"
# the token is read by the server from $WLK_API_TOKEN (kept out of the process list)
export WLK_API_TOKEN
# shellcheck disable=SC2086
exec .venv/bin/rt-translate-server "$@" $EXTRA_ARGS
