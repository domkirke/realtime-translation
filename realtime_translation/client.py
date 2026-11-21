"""
Real-time transcription + translation client for a WhisperLiveKit server.

    audio device (soundcard) / file --PCM s16le 16 kHz--> WhisperLiveKit server (WebSocket /asr)
                                     <-------- JSON ------  (Voxtral MLX / mlx-whisper, NLLW translation)
    --> stdout, OSC (/transcription_out, /translation_out, /transcription_line), JSONL file

By default the server is launched as a subprocess (realtime_translation.server, i.e. whisperlivekit-server
with the MLX single-thread fix) and stopped on exit.

Examples :
    rt-translate --device CLOUD_OUT                              # Voxtral MLX, auto language
    rt-translate --device CLOUD_OUT --lan fr --target-language en
    rt-translate --backend mlx-whisper --model small --lan auto
    rt-translate --file speech.mp3 --target-language fr          # stream a file (test)
    rt-translate --server-url wss://my-gpu-server/asr --token XXX --session-target-language fr
    rt-translate --list-devices
"""
import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import websockets

SAMPLE_RATE = 16000
MAX_OSC_STR_LENGTH = 9000  # same split as the CLOUD server (/transcription_out 1 / 2)
DEFAULT_LOG = Path.home() / ".cache" / "realtime-translation" / "server.log"

log = logging.getLogger("rt-translate")


# ---------------------------------------------------------------------------------------------
# WhisperLiveKit server process

def find_server_command():
    """the server runs in the same environment as this client, with the MLX single-thread fix."""
    return [sys.executable, "-m", "realtime_translation.server"]


def launch_server(args):
    cmd = [*find_server_command(),
           "--host", args.host, "--port", str(args.port),
           "--backend", args.backend,
           "--lan", args.lan,
           "--pcm-input"]
    if args.model:
        cmd += ["--model", args.model]
    if args.target_language:
        cmd += ["--target-language", args.target_language]
    if args.target_language or args.session_target_language:
        cmd += ["--nllb-size", args.nllb_size, "--nllb-backend", args.nllb_backend,
                "--translation-device", args.translation_device]
    if args.diarization:
        cmd += ["--diarization"]
    if args.token:
        cmd += ["--api-token", args.token]
    cmd += args.server_args
    log.info("launching : %s", " ".join(cmd))
    log_path = Path(args.server_log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    log.info("server log : %s", log_path)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    # model download + warmup can take a while the first time
    health = f"http://{args.host}:{args.port}/health"
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError("whisperlivekit-server exited with code %s (see %s)" % (proc.returncode, log_path))
        try:
            with urllib.request.urlopen(health, timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        if time.time() - t0 > args.server_timeout:
            proc.terminate()
            raise RuntimeError("whisperlivekit-server not ready after %ds" % args.server_timeout)
        time.sleep(0.5)
    log.info("server ready (%.1fs)", time.time() - t0)
    return proc


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------
# audio sources : yield float32 mono blocks at 16 kHz, pushed into an asyncio queue from a thread

def to_pcm16(block):
    block = np.clip(np.asarray(block, dtype=np.float32), -1., 1.)
    return (block * 32767).astype("<i2").tobytes()


def device_source(name, block_sec, stop_event):
    import soundcard as sc
    if name is None:
        mic = sc.default_microphone()
    else:
        mic = sc.get_microphone(name, include_loopback=(sys.platform != "darwin"))
    log.info("capturing from device : %s", mic.name)
    n = int(SAMPLE_RATE * block_sec)
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=min(n, 512)) as rec:
        while not stop_event.is_set():
            data = rec.record(numframes=n)
            yield data.mean(axis=1) if data.ndim > 1 else data


def file_source(path, block_sec, stop_event, realtime=True):
    import librosa
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    log.info("streaming file %s (%.1fs)", path, len(audio) / SAMPLE_RATE)
    n = int(SAMPLE_RATE * block_sec)
    t0 = time.time()
    for i in range(0, len(audio), n):
        if stop_event.is_set():
            break
        yield audio[i:i + n]
        if realtime:  # pace like a live input
            delay = t0 + (i + n) / SAMPLE_RATE - time.time()
            if delay > 0:
                time.sleep(delay)


def start_audio_thread(gen, loop, queue):
    def run():
        try:
            for block in gen:
                if loop.is_closed():
                    return
                loop.call_soon_threadsafe(queue.put_nowait, to_pcm16(block))
        except Exception as e:
            if not loop.is_closed():  # otherwise : normal shutdown
                log.error("audio source error : %s", e)
        finally:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, None)  # end of stream
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------------------------
# outputs

class Outputs(object):
    def __init__(self, args):
        self.osc = None
        if args.osc_port:
            from pythonosc.udp_client import SimpleUDPClient
            self.osc = SimpleUDPClient(args.osc_host, args.osc_port)
            log.info("OSC out -> %s:%d", args.osc_host, args.osc_port)
        self.jsonl = open(args.jsonl, "a") if args.jsonl else None
        self.quiet = args.quiet
        self.raw = args.raw
        self.translate = bool(args.target_language or args.session_target_language)
        self._live_lines = 0  # number of terminal lines of the live display currently drawn
        self._last_text = None
        self._last_trans = None
        self._n_committed = 0

    @staticmethod
    def _split_osc(text):
        return text[:MAX_OSC_STR_LENGTH], text[MAX_OSC_STR_LENGTH:]

    def __call__(self, msg):
        if self.raw:
            print(json.dumps(msg, ensure_ascii=False), flush=True)
        lines = [l for l in (msg.get("lines") or []) if l.get("text")]
        committed = " ".join(l["text"].strip() for l in lines)
        committed_tr = " ".join(l["translation"].strip() for l in lines if l.get("translation"))
        text = (committed + " " + (msg.get("buffer_transcription") or "")).strip()
        trans = (committed_tr + " " + (msg.get("buffer_translation") or "")).strip()

        text_changed = text != self._last_text
        if text_changed:
            self._last_text = text
            if self.osc:
                a, b = self._split_osc(text)
                self.osc.send_message("/transcription_out", (1, a))
                self.osc.send_message("/transcription_out", (2, b))
        trans_changed = bool(trans) and trans != self._last_trans
        if trans_changed:
            self._last_trans = trans
            if self.osc:
                a, b = self._split_osc(trans)
                self.osc.send_message("/translation_out", (1, a))
                self.osc.send_message("/translation_out", (2, b))

        # newly finalized lines : one OSC message + one JSONL record each
        for line in lines[self._n_committed:-1] if len(lines) > 1 else []:
            self._commit(line)
        n_before = self._n_committed
        self._n_committed = max(self._n_committed, len(lines) - 1)
        if (text_changed or trans_changed or self._n_committed > n_before) and not self.quiet and not self.raw:
            live = lines[self._n_committed:]
            live_text = " ".join(l["text"].strip() for l in live) + " " + (msg.get("buffer_transcription") or "")
            live_tr = " ".join(l["translation"].strip() for l in live if l.get("translation")) + " " \
                      + (msg.get("buffer_translation") or "")
            self._draw_live(live_text.strip(), live_tr.strip())

    # live display : the end of the current transcription (and translation) on fixed lines, redrawn in place

    def _clear_live(self):
        if self._live_lines:
            # back to the first line of the live block, then clear to the end of the screen
            print("\r" + "\033[F" * (self._live_lines - 1) + "\033[J", end="")
            self._live_lines = 0

    def _draw_live(self, text, trans):
        import shutil
        width = max(20, shutil.get_terminal_size((100, 20)).columns - 6)
        fit = lambda t: ("…" + t[-(width - 1):]) if len(t) > width else t  # one terminal line, no wrapping
        rows = ["  " + fit(text)]
        if self.translate:
            rows.append("\033[36m> " + fit(trans or "…") + "\033[0m")
        self._clear_live()
        print("\n".join(rows), end="", flush=True)
        self._live_lines = len(rows)

    def _commit(self, line):
        if self.osc:
            self.osc.send_message("/transcription_line", (line["text"].strip(), (line.get("translation") or "").strip(),
                                                          str(line.get("start", "")), str(line.get("end", ""))))
        if self.jsonl:
            self.jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.jsonl.flush()
        if not self.quiet and not self.raw:
            self._clear_live()
            tr = line.get("translation")
            print("[%s - %s] %s%s" % (line.get("start"), line.get("end"), line["text"].strip(),
                                      ("\n    \033[36m-> " + tr.strip() + "\033[0m") if tr else ""), flush=True)

    def finish(self, msg):
        lines = [l for l in ((msg or {}).get("lines") or []) if l.get("text")]
        if not self.quiet and not self.raw:
            self._clear_live()
        for line in lines[self._n_committed:]:
            self._commit(line)
        if self.jsonl:
            self.jsonl.close()


# ---------------------------------------------------------------------------------------------
# websocket client

async def run_client(args, stop_event):
    params = {}
    if args.session_language:
        params["language"] = args.session_language
    if args.session_target_language:
        params["target_language"] = args.session_target_language
    url = args.server_url or f"ws://{args.host}:{args.port}/asr"
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    if args.file:
        gen = file_source(args.file, args.block, stop_event, realtime=not args.fast)
    else:
        gen = device_source(args.device, args.block, stop_event)
    outputs = Outputs(args)
    last_msg = None

    log.info("connecting to %s", url)
    headers = {"Authorization": "Bearer " + args.token} if args.token else None
    async with websockets.connect(url, max_size=None, open_timeout=args.open_timeout, ping_timeout=None,
                                  additional_headers=headers) as ws:
        async def sender():
            start_audio_thread(gen, loop, queue)
            while True:
                chunk = await queue.get()
                if chunk is None:
                    await ws.send(b"")  # tells the server the stream is over
                    return
                await ws.send(chunk)

        send_task = asyncio.create_task(sender())
        try:
            last_rx = time.time()
            while True:
                # once all audio is sent, don't wait forever if the server never sends ready_to_stop
                try:
                    raw = await asyncio.wait_for(ws.recv(), 1.0)
                    last_rx = time.time()
                except asyncio.TimeoutError:
                    if send_task.done() and time.time() - last_rx > args.end_timeout:
                        log.warning("no answer from the server %ss after the end of the stream, closing",
                                    args.end_timeout)
                        break
                    continue
                except websockets.ConnectionClosed:
                    break
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "config":
                    if not msg.get("useAudioWorklet"):
                        log.warning("server was not started with --pcm-input : raw PCM will be misread")
                    continue
                if t == "ready_to_stop":
                    break
                if t == "error" or msg.get("error"):
                    log.error("server error : %s", msg.get("error"))
                    if t == "error":
                        break
                last_msg = msg
                outputs(msg)
        finally:
            send_task.cancel()
            outputs.finish(last_msg)
            print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("server")
    g.add_argument("--no-launch", dest="launch", action="store_false", help="connect to an already running server")
    g.add_argument("--server-url", default=None, help="websocket url (default ws://HOST:PORT/asr)")
    g.add_argument("--token", default=os.environ.get("RT_TRANSLATE_TOKEN"),
                   help="API token of a remote server (default : $RT_TRANSLATE_TOKEN) ; with a launched server, "
                        "also protects it")
    g.add_argument("--host", default="localhost")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--backend", default="voxtral-mlx",
                   help="voxtral-mlx (recommended on Apple Silicon), mlx-whisper, faster-whisper, ...")
    g.add_argument("--model", default=None, help="whisper size for whisper backends (tiny/base/small/medium/large-v3...)")
    g.add_argument("--lan", default="auto", help="source language (fr, en, ..., auto)")
    g.add_argument("--target-language", default=None, help="translation target via NLLW (fr, en, uk, ru, ...)")
    g.add_argument("--nllb-size", default="600M", choices=["600M", "1.3B"],
                   help="NLLB translation model : 600M (2.5 GB, faster) or 1.3B (5.5 GB, better)")
    g.add_argument("--nllb-backend", default="transformers", choices=["transformers", "ctranslate2"])
    g.add_argument("--translation-device", default="cpu", choices=["auto", "mps", "cpu"],
                   help="where the NLLB model runs. mps (Apple GPU) crashes with torch 2.15 on macOS 14")
    g.add_argument("--diarization", action="store_true")
    g.add_argument("--server-log", default=str(DEFAULT_LOG), help="log file of the launched server")
    g.add_argument("--server-timeout", type=float, default=900, help="seconds to wait for model download/warmup")
    g.add_argument("--open-timeout", type=float, default=600,
                   help="websocket handshake timeout (the first session may load/download the translation model)")
    g.add_argument("--end-timeout", type=float, default=30,
                   help="seconds to wait for the last results once the audio stream is over")
    g.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                   help="extra args passed verbatim to whisperlivekit-server (must be last)")
    g = p.add_argument_group("session (per-connection overrides, server must support them)")
    g.add_argument("--session-language", default=None)
    g.add_argument("--session-target-language", default=None)
    g = p.add_argument_group("input")
    g.add_argument("--device", default=None, help="capture device name (soundcard), e.g. CLOUD_OUT ; default mic otherwise")
    g.add_argument("--file", default=None, help="stream an audio file instead of a device")
    g.add_argument("--fast", action="store_true", help="with --file : send as fast as possible, not in real time")
    g.add_argument("--block", type=float, default=0.1, help="audio block size in seconds")
    g.add_argument("--list-devices", action="store_true")
    g = p.add_argument_group("output")
    g.add_argument("--osc-host", default="127.0.0.1")
    g.add_argument("--osc-port", type=int, default=8889, help="0 to disable OSC (8889 = CLOUD transcription out port)")
    g.add_argument("--jsonl", default=None, help="append finalized lines to this JSONL file")
    g.add_argument("--raw", action="store_true", help="print every raw JSON message from the server")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s : %(message)s", stream=sys.stderr)

    if args.list_devices:
        import soundcard as sc
        for m in sc.all_microphones(include_loopback=(sys.platform != "darwin")):
            print(m.name)
        return

    def _on_sigterm(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)  # so the server is stopped too

    proc = None
    stop_event = threading.Event()
    try:
        if args.launch and not args.server_url:
            proc = launch_server(args)
        asyncio.run(run_client(args, stop_event))
    except KeyboardInterrupt:
        pass
    except websockets.exceptions.InvalidStatus as e:
        code = e.response.status_code
        hint = " : wrong or missing --token ?" if code in (401, 403) else ""
        log.error("server refused the connection (HTTP %d)%s", code, hint)
        sys.exit(1)
    except (OSError, websockets.exceptions.InvalidHandshake) as e:
        log.error("cannot connect to the server : %s", e)
        sys.exit(1)
    finally:
        stop_event.set()
        stop_server(proc)


if __name__ == "__main__":
    main()
