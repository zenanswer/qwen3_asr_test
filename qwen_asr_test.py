import pyaudio
import numpy as np
import threading
import queue
import time
import torch

from qwen_asr import Qwen3ASRModel

MODEL_PATH = "./models/Qwen/Qwen3-ASR-1.7B"
LANGUAGE = "English"

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

RATE = 16000
CHUNK = 1600
INFERENCE_WINDOW_SECONDS = 5
INFERENCE_WINDOW_SAMPLES = RATE * INFERENCE_WINDOW_SECONDS
CHUNK_COUNT_PER_INFER = INFERENCE_WINDOW_SECONDS * 10

print(f"[INFO] Device: {DEVICE}, dtype: {DTYPE}")
print(f"[INFO] Loading model from {MODEL_PATH}...")

model = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    device_map=DEVICE,
    max_new_tokens=256,
)

print("▶ Qwen3-ASR 实时识别已启动\n")

audio_buf = np.array([], dtype=np.float32)
stable_text = ""
result_queue = queue.Queue()
inference_done = threading.Event()
inference_done.set()

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

next_audio_start = 0


def infer(audio_segment, segment_idx):
    inference_done.wait()
    inference_done.clear()
    t0 = time.time()
    results = model.transcribe(
        audio=(audio_segment, RATE),
        language=LANGUAGE,
    )
    infer_time = time.time() - t0
    text = results[0].text.strip() if results else ""
    result_queue.put((text, infer_time, segment_idx))
    inference_done.set()


chunk_count = 0
last_infer_count = 0

try:
    while True:
        chunk = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buf = np.concatenate([audio_buf, data])

        chunk_count += 1

        chunks_since_last_infer = chunk_count - last_infer_count

        if (
            chunks_since_last_infer >= CHUNK_COUNT_PER_INFER
            and inference_done.is_set()
            and len(audio_buf) >= next_audio_start + INFERENCE_WINDOW_SAMPLES
        ):
            last_infer_count = chunk_count
            audio_segment = audio_buf[
                next_audio_start : next_audio_start + INFERENCE_WINDOW_SAMPLES
            ].copy()
            threading.Thread(
                target=infer,
                args=(audio_segment, next_audio_start // INFERENCE_WINDOW_SAMPLES),
                daemon=True,
            ).start()
            next_audio_start += INFERENCE_WINDOW_SAMPLES

        try:
            text, infer_time, segment_idx = result_queue.get_nowait()
            if text:
                stable_text = text
                print(
                    f"\n[DEBUG] transcribe done [{segment_idx}]: '{text}' | infer_time={infer_time:.3f}s",
                    flush=True,
                )
        except queue.Empty:
            pass

        if chunk_count % 50 == 0:
            print(
                f"\n[DEBUG] loop alive, buf_len={len(audio_buf)}, chunk={chunk_count}, next_audio_start={next_audio_start}",
                flush=True,
            )

        if stable_text:
            print(f"\r{stable_text}", end="", flush=True)

except KeyboardInterrupt:
    print("\n\n✅ 结束")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
