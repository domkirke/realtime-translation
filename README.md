# realtime-translation

Local, low-latency real-time transcription and translation, built on
[WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit):

- audio from a capture device (or a file) is streamed over WebSocket to a local WhisperLiveKit server
- text comes back as a stream of JSON messages
- the ASR backend is Voxtral MLX by default (native on Apple Silicon); mlx-whisper and faster-whisper also work
- translation uses NLLW (NLLB, 200 languages)
- results go to the terminal, OSC and/or a JSONL file

## Install

```sh
./install.sh
```

The script creates `.venv/` in this folder (with `uv` if available, otherwise `venv` + `pip`) and installs the
package in editable mode. On Apple Silicon it also installs the MLX extras. The models are downloaded on the first
run: Voxtral is about 3 GB and NLLB 600M about 2.5 GB.

## Usage

```sh
.venv/bin/rt-translate --list-devices
.venv/bin/rt-translate --device "CLOUD_OUT"                                 # transcription, language auto-detected
.venv/bin/rt-translate --device "CLOUD_OUT" --lan fr --target-language en   # + translation
.venv/bin/rt-translate --backend mlx-whisper --model small                  # Whisper instead of Voxtral
.venv/bin/rt-translate --file speech.mp3 --target-language uk               # stream a file in real time
.venv/bin/rt-translate --no-launch --server-url ws://host:8000/asr          # use a server that is already running
.venv/bin/rt-translate ... --server-args --nllb-size 1.3B                   # extra args go to the server (must be last)
```

By default `rt-translate` starts the server itself (log in `~/.cache/realtime-translation/server.log`) and stops it on
exit. To run the server on its own, use `rt-translate-server`. It takes the same arguments as `whisperlivekit-server`;
add `--pcm-input` so that `rt-translate` clients can connect to it.

## Outputs

| output | content |
|---|---|
| OSC `/transcription_out (1, text) (2, overflow)` | full transcription so far (committed + live buffer), split at 9000 characters |
| OSC `/translation_out (1, text) (2, overflow)` | full translation so far |
| OSC `/transcription_line (text, translation, start, end)` | one message per finalized segment |
| `--jsonl FILE` | one JSON object per finalized segment |
| `--raw` | every raw JSON message from the server, printed to stdout |

OSC is sent to `127.0.0.1:8889` by default (`--osc-host`, `--osc-port`; `--osc-port 0` turns it off).

## Notes

- `rt-translate-server` is `whisperlivekit-server` with one fix. Recent MLX versions bind GPU streams to the thread
  that created them, but WhisperLiveKit runs each ASR call on an arbitrary pool thread. That makes the MLX backends
  crash at random with `There is no Stream(gpu, N) in current thread`. The wrapper runs every ASR call on a single
  dedicated thread and patches this at runtime, so the installed package is not modified.
- The first connection with translation enabled can take a while, because the translation model is loaded at that
  point. `--open-timeout` defaults to 600 s for this reason.
- Translation model: `--nllb-size 600M` (default) keeps up live on a Mac CPU. `--nllb-size 1.3B` translates a
  little better, but on a Mac it runs on the CPU and is far too slow for live use: a 17 s clip took 216 s. Use it on
  a GPU server (see below). The Apple GPU option (`--translation-device mps`) crashes with torch 2.15 on macOS 14.

## Running on a remote GPU

See [deploy/README.md](deploy/README.md): one 24 GB NVIDIA GPU runs Voxtral and NLLB 1.3B, and
`rt-translate --server-url ... --token ...` connects to it from your computer.
