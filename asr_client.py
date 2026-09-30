import pyaudio
import numpy as np
import requests
import threading
import time
import wave
import os

SERVER_URL = "http://10.10.11.32:8001"
RATE = 16000
CHUNK = 1600
CHUNK_MS = 100
MAX_CLIENT_AUDIO_SECONDS = 10
MAX_CLIENT_SAMPLES = RATE * MAX_CLIENT_AUDIO_SECONDS

session_id = int(time.time() * 1000) % 1000000

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

streaming = True
audio_buffer = np.array([], dtype=np.int16)
all_results = []

os.makedirs("./client_recordings", exist_ok=True)


def send_audio():
    global streaming
    send_count = 0
    client_audio_list = []
    max_retries = 3
    retry_count = 0

    while streaming:
        if retry_count >= max_retries:
            print("[CLIENT] Max retries reached, stopping...", flush=True)
            streaming = False
            break

        chunk = stream.read(CHUNK, exception_on_overflow=False)
        chunk_np = np.frombuffer(chunk, dtype=np.int16)
        client_audio_list.append(chunk_np)

        send_count += 1
        try:
            response = requests.post(
                f"{SERVER_URL}/audio/{session_id}",
                data=chunk,
                timeout=5,
            )
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                for text in results:
                    if text not in all_results:
                        all_results.append(text)
                        print(f"\n[SENTENCE] {text}", flush=True)
                retry_count = 0
            else:
                retry_count += 1
                print(
                    f"[CLIENT] Server error {response.status_code}, retrying ({retry_count}/{max_retries})...",
                    flush=True,
                )
                time.sleep(0.1)
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(
                    f"[CLIENT] Error: {e}, retrying ({retry_count}/{max_retries})...",
                    flush=True,
                )
                time.sleep(0.1)
            else:
                print(f"[CLIENT] Error: {e}, max retries reached", flush=True)

        if send_count % 10 == 0:
            client_audio = np.concatenate(client_audio_list)
            client_wav_path = (
                f"./client_recordings/{session_id}_c{send_count // 10}.wav"
            )
            with wave.open(client_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(client_audio.tobytes())
            # print(
            #    f"[CLIENT] Saved client recording to {client_wav_path} | samples={len(client_audio)}",
            #    flush=True,
            # )

        # time.sleep(CHUNK_MS / 1000.0)

    stream.stop_stream()
    stream.close()
    p.terminate()


def poll_results():
    global streaming
    while streaming:
        time.sleep(1)
        try:
            response = requests.get(
                f"{SERVER_URL}/results/{session_id}",
                timeout=5,
            )
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                for text in results:
                    is_duplicate = any(
                        text == existing or text in existing
                        for existing in all_results[-3:]
                    )
                    if not is_duplicate and len(text) > 3:
                        all_results.append(text)
                        print(f"\n[SENTENCE] {text}")
        except Exception:
            pass


print(f"[CLIENT] Session {session_id}, starting...")
print("[CLIENT] Speak now...")

sender_thread = threading.Thread(target=send_audio, daemon=True)
poller_thread = threading.Thread(target=poll_results, daemon=True)

sender_thread.start()
poller_thread.start()

try:
    while streaming:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n[CLIENT] Shutting down...")
finally:
    streaming = False
    sender_thread.join(timeout=1)
    poller_thread.join(timeout=1)
