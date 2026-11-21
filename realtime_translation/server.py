"""
Launcher for whisperlivekit-server with two runtime patches (the installed packages are not modified) :

1. ASR calls run on ONE dedicated thread.
   Recent MLX versions bind GPU streams to the thread that created them. WhisperLiveKit offloads ASR
   calls with asyncio.to_thread, i.e. on any thread of the default pool, which makes the voxtral-mlx /
   mlx-whisper backends crash randomly with "RuntimeError: There is no Stream(gpu, N) in current thread".

2. The NLLB translation model can run on the Apple GPU (MPS, float16).
   NLLW only knows "cuda" or "cpu", so on a Mac the model runs on the CPU in float32, which is slow
   for the 1.3B model.

Takes the same arguments as whisperlivekit-server, plus :
    --translation-device {cpu,mps,auto}   (default cpu ; mps crashes with torch 2.15 on macOS 14)
Example :
    rt-translate-server --backend voxtral-mlx --pcm-input --lan auto --target-language en --nllb-size 1.3B
"""
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from whisperlivekit import audio_processor as _ap

log = logging.getLogger("rt-translate-server")


# -- 1. MLX : single ASR thread ---------------------------------------------------------------

_ASR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wlk-mlx")


async def _run_counted_transcription_call(self, method, *args):
    loop = asyncio.get_running_loop()
    _t0 = perf_counter()
    tokens, processed_upto = await loop.run_in_executor(_ASR_EXECUTOR, lambda: method(*args))
    self.metrics.transcription_durations.append(perf_counter() - _t0)
    self.metrics.n_transcription_calls += 1
    tokens = tokens or []
    self.metrics.n_tokens_produced += len(tokens)
    return tokens, processed_upto


# -- 2. NLLB on MPS ---------------------------------------------------------------------------

def _patch_translation_device(device):
    import torch
    import nllw.core as nc

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        return
    if device == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS is not available, translation stays on the CPU")
        return

    original_load_model = nc.load_model
    original_init = nc.TranslationBackend.__init__

    def load_model(*args, **kwargs):
        tm = original_load_model(*args, **kwargs)
        if tm.backend_type == "transformers":
            t0 = perf_counter()
            tm.translator = tm.translator.to(device, dtype=torch.float16)
            tm.device = device
            log.info("translation model moved to %s (float16) in %.1fs", device, perf_counter() - t0)
        return tm

    def init(self, *args, model=None, **kwargs):
        on_device = model is not None and not hasattr(model, "translate_batch") \
            and str(getattr(model, "device", "")).startswith(device)
        if not on_device:
            return original_init(self, *args, model=model, **kwargs)
        # NLLW would move the shared model back to "cpu" : neutralize .to() during its __init__
        model.to = lambda *a, **k: model
        try:
            original_init(self, *args, model=model, **kwargs)
        finally:
            del model.to  # removes the instance attribute, the class method is back
        self.device = device
        self.stable_prefix_tokens = self.stable_prefix_tokens.to(device)

    nc.load_model = load_model
    nc.TranslationBackend.__init__ = init
    # whisperlivekit does `from nllw import load_model`
    import nllw
    nllw.load_model = load_model


def _pop_option(argv, name, default):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if a.startswith(name + "="):
            del argv[i]
            return a.split("=", 1)[1]
    return default


def main():
    logging.basicConfig(level=logging.INFO)
    # "cpu" = leave NLLW alone : CPU on a Mac, CUDA automatically on an NVIDIA server
    translation_device = _pop_option(sys.argv, "--translation-device", "cpu")
    if sys.platform == "darwin":
        # MLX only ; on a CUDA server, sessions keep running their ASR calls in parallel
        _ap.AudioProcessor._run_counted_transcription_call = _run_counted_transcription_call
    _patch_translation_device(translation_device)
    from whisperlivekit.basic_server import main as server_main
    server_main()


if __name__ == "__main__":
    main()
