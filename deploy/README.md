# Running the server on a GPU machine

Speech recognition and translation run on a rented NVIDIA GPU. `rt-translate` on your own computer still captures
the audio device and sends the results to OSC / JSONL / the terminal; it simply connects to the remote server
instead of starting a local one.

```
Mac : audio device -> rt-translate --server-url ... --token ...  ==(WebSocket)==>  GPU server : Voxtral + NLLB 1.3B
       OSC / JSONL / terminal <-----------------------------------------------------  JSON results
```

## Which GPU

One GPU is enough. These are estimates for the default setup:

| component | VRAM |
|---|---|
| Voxtral Mini 4B realtime (`--backend voxtral`, bf16) | ~9–10 GB |
| NLLB 1.3B translation (fp32) | ~5.5 GB |
| working memory | 1–3 GB |

- **24 GB card (L4, A10G, RTX 3090 / 4090, A5000)**: the default setup, with room to spare.
- **16 GB card (T4)**: use `BACKEND=faster-whisper MODEL=large-v3-turbo` instead of Voxtral.
- Disk: count about 40 GB. The models take ~15 GB and are downloaded on the first start (cached afterwards).

These numbers have not been measured on a GPU yet. Check `nvidia-smi` on the first run.

## Option A: a rented GPU machine with SSH (simplest)

This works with RunPod, Lambda, Vast.ai or a cloud VM: any Linux machine with an NVIDIA driver that you can reach
with SSH.

1. Copy the project (without the local `.venv`):
   ```sh
   rsync -av --exclude .venv --exclude __pycache__ ~/Dropbox/code/realtime-translation/ USER@HOST:realtime-translation/
   ```
2. On the server, inside `tmux` so it keeps running when you disconnect:
   ```sh
   cd realtime-translation
   export WLK_API_TOKEN=$(openssl rand -hex 24); echo $WLK_API_TOKEN    # keep this value
   ./deploy/server.sh
   ```
   The first run installs everything and downloads the models; it is ready when it prints `Uvicorn running`.
   Settings are passed as environment variables, for example
   `TARGET_LANGUAGE=fr LAN=en NLLB_SIZE=1.3B ./deploy/server.sh` (see the top of `server.sh`).
3. On your Mac, open an SSH tunnel. The server only listens on the machine itself, so the tunnel is what makes it
   reachable, and it is encrypted:
   ```sh
   ssh -N -L 8000:127.0.0.1:8000 USER@HOST
   ```
4. In another terminal on your Mac:
   ```sh
   export RT_TRANSLATE_TOKEN=<the token>
   .venv/bin/rt-translate --server-url ws://localhost:8000/asr --device CLOUD_OUT
   .venv/bin/rt-translate --server-url ws://localhost:8000/asr --device CLOUD_OUT \
       --session-language en --session-target-language fr     # override the languages for this session
   ```

Stop the machine when you are done: GPU rental is billed by the hour.

## Option B: Docker (your own GPU server)

This needs the NVIDIA driver and the NVIDIA Container Toolkit on the host.

```sh
export WLK_API_TOKEN=$(openssl rand -hex 24)
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f
```

The port is published on `127.0.0.1:8000` only. Reach it with the same SSH tunnel as in option A. The models are
kept in the `models` Docker volume.

## Exposing it publicly (without a tunnel)

Run with `HOST=0.0.0.0` behind an HTTPS reverse proxy, for example the HTTPS proxy of your GPU provider, or Caddy
or nginx with a certificate. Then connect with `--server-url wss://YOUR-DOMAIN/asr`.

- Never expose it as plain `ws://` on the internet: the audio and the token would travel unencrypted.
- Always set a token. `server.sh` refuses to start without one unless `ALLOW_NO_TOKEN=1` is set.

## Notes

- One server can serve several clients at once, and each connection is an independent session. How many
  simultaneous streams a single GPU sustains has not been measured.
- `--session-target-language` only works if the server was started with translation, i.e. a non-empty
  `TARGET_LANGUAGE`.
