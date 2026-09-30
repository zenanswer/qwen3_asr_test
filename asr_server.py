import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from qwen_asr import Qwen3ASRModel
import threading
import queue
import time
import wave
import os

# MODEL_PATH = "./models/Qwen/Qwen3-ASR-1.7B"
MODEL_PATH = "./models/Qwen3-ASR-0.6B"
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
CHUNK_SAMPLES = 480
VAD_WINDOW = 512
# LOOKBACK_CHUNKS = 3  # Removed to prevent audio overlap (AABBCC issue)
SILENCE_CHUNKS = 5
VAD_THRESHOLD = 0.5
AMP_THRESHOLD = 0.005
MAX_SESSION_AGE = 60
MAX_SESSION_SAMPLES = RATE * 60
MIN_SPEECH_SAMPLES = int(RATE * 0.3)

os.makedirs("./audio_segments", exist_ok=True)

print(f"[INFO] Device: {DEVICE}, dtype: {DTYPE}")
print(f"[INFO] Loading model from {MODEL_PATH}...")

model = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    device_map=DEVICE,
    max_new_tokens=256,
)
print("[INFO] Model loaded successfully")

print("[INFO] Loading Silero VAD...")
try:
    from silero_vad import load_silero_vad

    vad_model = load_silero_vad()
    vad_model.eval()
    print("[INFO] Silero VAD loaded successfully")
except Exception as e:
    print(f"[WARN] Failed to load Silero VAD: {e}")
    vad_model = None

app = FastAPI(title="Qwen3-ASR Server")

sessions = {}
infer_queue = queue.Queue()
infer_thread_running = True
pending_segments = {}
results_lock = threading.Lock()


def infer_worker():
    while infer_thread_running:
        try:
            item = infer_queue.get(timeout=0.05)
            session_id, audio, segment_key = item[:3]
            audio_path = item[3] if len(item) > 3 else None

            print(
                f"[INFER] Processing | session={session_id} | audio_samples={len(audio)}"
            )

            if session_id not in sessions:
                print(f"[INFER] Session not found: {session_id}")
                pending_segments.pop(segment_key, None)
                continue

            try:
                results = model.transcribe(
                    audio=(audio, RATE),
                    language=LANGUAGE,
                )
                text = results[0].text.strip() if results else ""
                print(f"[INFER] Result: '{text}'")

                if text:
                    with results_lock:
                        if session_id in sessions:
                            sessions[session_id]["results"].append(text)
                            sessions[session_id]["last_update"] = time.time()
                            print(f"[INFER] Added to session {session_id}")
            except Exception as e:
                print(f"[SERVER] Infer error: {e}")
            finally:
                pending_segments.pop(segment_key, None)

        except queue.Empty:
            continue


infer_thread = threading.Thread(target=infer_worker, daemon=True)
infer_thread.start()


def cleanup_old_sessions():
    now = time.time()
    to_delete = []
    for sid, sess in sessions.items():
        if now - sess["last_update"] > MAX_SESSION_AGE:
            to_delete.append(sid)
    for sid in to_delete:
        del sessions[sid]
        print(f"[CLEANUP] Deleted old session {sid}")


@app.post("/audio/{session_id}")
async def receive_audio(session_id: int, request: Request):
    cleanup_old_sessions()

    body = await request.body()
    if not body or len(body) == 0:
        raise HTTPException(status_code=400, detail="No audio data")

    audio_data = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_samples = len(audio_data)

    now = time.time()
    if session_id not in sessions:
        sessions[session_id] = {
            "audio": np.array([], dtype=np.float32),
            "results": [],
            "speech_start": None,
            "consecutive_silence": 0,
            "last_update": now,
            "prev_sentence_end": 0,
        }
        print(f"[SESSION] Created session {session_id}")

    session = sessions[session_id]
    session["last_update"] = now

    buffer_len_before = len(session["audio"])
    session["audio"] = np.concatenate([session["audio"], audio_data])

    last_512 = (
        session["audio"][-VAD_WINDOW:]
        if len(session["audio"]) >= VAD_WINDOW
        else session["audio"]
    )
    if len(last_512) < VAD_WINDOW:
        last_512 = np.pad(last_512, (0, VAD_WINDOW - len(last_512)), mode="constant")
    audio_tensor = torch.from_numpy(last_512).float()

    with torch.no_grad():
        if vad_model is not None:
            vad_out = vad_model(audio_tensor, RATE)
            vad_prob = (
                vad_out["speech"].item()
                if isinstance(vad_out, dict)
                else vad_out.item()
            )
        else:
            vad_prob = 0.5

    amplitude = float(np.max(np.abs(audio_data)) if chunk_samples > 0 else 0.0)
    has_sound = amplitude > AMP_THRESHOLD

    is_speech = bool(vad_prob > VAD_THRESHOLD) and has_sound

    if is_speech:
        print(
            f"[VAD] Speech | prob={vad_prob:.4f} | amp={amplitude:.4f} | session={session_id}"
        )
        if session["speech_start"] is None:
            session["speech_start"] = buffer_len_before
        session["consecutive_silence"] = 0
    else:
        session["consecutive_silence"] += 1
        print(
            f"[VAD] Silence | prob={vad_prob:.4f} | amp={amplitude:.4f} | consecutive={session['consecutive_silence']}/{SILENCE_CHUNKS} | session={session_id}"
        )

        if (
            session["consecutive_silence"] >= SILENCE_CHUNKS
            and session["speech_start"] is not None
        ):
            # --- FIXED: Removed lookback to prevent AABBCC overlap ---
            seg_start = session["prev_sentence_end"]

            # Safety check to prevent index out of bounds
            if seg_start >= len(session["audio"]):
                print(f"[WARN] Invalid seg_start {seg_start}, resetting state.")
                session["speech_start"] = None
                session["consecutive_silence"] = 0
                return {
                    "results": [str(r) for r in session["results"]],
                    "vad_prob": float(vad_prob),
                    "speech": bool(is_speech),
                }

            segment_audio = session["audio"][seg_start:].copy()

            segment_key = (session_id, seg_start, len(segment_audio))
            if segment_key not in pending_segments:
                pending_segments[segment_key] = True

                timestamp = int(time.time() * 1000)
                audio_path = f"./audio_segments/{session_id}_{timestamp}.wav"
                with wave.open(audio_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes((segment_audio * 32768).astype(np.int16).tobytes())
                print(
                    f"[VAD] Sentence ended | seg_start={seg_start} | seg_samples={len(segment_audio)} | session={session_id}"
                )

                infer_queue.put((session_id, segment_audio, segment_key, audio_path))

                # Update the end pointer to the current total length
                session["prev_sentence_end"] = len(session["audio"])
                # Do NOT clear the audio buffer; let trim handle it.
                # session["audio"] = np.array([], dtype=np.float32) # <-- REMOVED

                session["speech_start"] = None
                session["consecutive_silence"] = 0

    # Trim the audio buffer if it's too long
    if len(session["audio"]) > MAX_SESSION_SAMPLES:
        trim_start = len(session["audio"]) - MAX_SESSION_SAMPLES
        session["audio"] = session["audio"][trim_start:]
        # Adjust pointers after trimming
        session["prev_sentence_end"] = max(0, session["prev_sentence_end"] - trim_start)
        if session["speech_start"] is not None:
            session["speech_start"] = max(0, session["speech_start"] - trim_start)

    return {
        "results": [str(r) for r in session["results"]],
        "vad_prob": float(vad_prob),
        "speech": bool(is_speech),
    }


@app.get("/results/{session_id}")
async def get_results(session_id: int):
    cleanup_old_sessions()
    if session_id not in sessions:
        return {"results": []}
    with results_lock:
        return {"results": [str(r) for r in sessions[session_id]["results"]]}


@app.delete("/session/{session_id}")
async def delete_session(session_id: int):
    if session_id in sessions:
        del sessions[session_id]
        print(f"[SESSION] Deleted session {session_id}")
    return {"status": "ok"}


@app.on_event("shutdown")
def shutdown_event():
    global infer_thread_running
    infer_thread_running = False
    print("[SERVER] Shutting down")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
