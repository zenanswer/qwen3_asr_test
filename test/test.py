import io
import os
import queue
import threading
import time
import wave

import numpy as np
import pyaudio
import requests

RATE = 16000
CHUNK = 1600
SERVER_ASR = "http://10.10.11.32:8001"
SERVER_TTS = "http://10.10.11.32:8002"


class CombinedClient:
    def __init__(
        self,
        asr_url: str = SERVER_ASR,
        tts_url: str = SERVER_TTS,
    ):
        self.asr_url = asr_url
        self.tts_url = tts_url
        self.session_id = int(time.time() * 1000) % 1000000

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

        os.makedirs("./test_recordings", exist_ok=True)

    def _init_stream(self):
        if self.stream is None:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

    def _close_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def speak_and_real_time_recognize(
        self,
        welcome_text: str = "欢迎使用，请说话",
        record_duration: float = 5.0,
    ) -> dict:
        print(f"[CLIENT] Step 1: Calling TTS for welcome message...")
        tts_audio = self._call_tts(welcome_text)
        print(f"[CLIENT] TTS audio received, size: {len(tts_audio)} bytes")

        print(f"[CLIENT] Step 2: Playing TTS and recording from mic...")
        recorded_audio = self._play_and_record(tts_audio, record_duration)
        print(f"[CLIENT] Recording done, samples: {len(recorded_audio)}")

        print(f"[CLIENT] Step 3: Saving recording...")
        record_path = self._save_recording(recorded_audio)

        print(f"[CLIENT] Step 4: Sending to ASR...")
        asr_results = self._send_to_asr(recorded_audio)

        return {
            "asr_results": asr_results,
            "recorded_file": record_path,
        }

    def _call_tts(self, text: str) -> bytes:
        response = requests.post(
            f"{self.tts_url}/tts",
            json={
                "text": text,
                "voice_description": "女声，语速中等",
                "language": "auto",
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(f"TTS failed: {response.status_code} - {response.text}")
        return response.content

    def _play_and_record(
        self, audio_bytes: bytes, record_duration: float
    ) -> np.ndarray:
        import soundfile as sf

        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        audio_int16 = (audio_data * 32768).astype(np.int16)

        self._init_stream()
        self.is_recording = True

        record_frames = int(RATE / CHUNK * record_duration)
        recorded_frames = []

        def play_audio():
            try:
                self.stream.write(audio_int16.tobytes())
            except Exception as e:
                print(f"[CLIENT] Playback error: {e}")

        play_thread = threading.Thread(target=play_audio)
        play_thread.start()

        for _ in range(record_frames):
            if not self.is_recording:
                break
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                recorded_frames.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                print(f"[CLIENT] Recording error: {e}")
                break

        play_thread.join(timeout=1)
        self.is_recording = False

        if recorded_frames:
            return np.concatenate(recorded_frames)
        return np.array([], dtype=np.int16)

    def _save_recording(self, audio: np.ndarray) -> str:
        timestamp = int(time.time() * 1000)
        path = f"./test_recordings/{self.session_id}_{timestamp}.wav"
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(audio.tobytes())
        print(f"[CLIENT] Recording saved to {path}")
        return path

    def _send_to_asr(self, audio: np.ndarray) -> list:
        audio_float32 = audio.astype(np.float32) / 32768.0

        response = requests.post(
            f"{self.asr_url}/asr/audio/{self.session_id}",
            data=audio.tobytes(),
            timeout=30,
        )

        if response.status_code != 200:
            print(f"[CLIENT] ASR error: {response.status_code}")
            return []

        result = response.json()
        results = result.get("results", [])

        for i, text in enumerate(results):
            print(f"[CLIENT] ASR Result {i + 1}: {text}")

        return results

    def get_asr_results(self) -> list:
        response = requests.get(
            f"{self.asr_url}/asr/results/{self.session_id}",
            timeout=5,
        )
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def close(self):
        self._close_stream()
        self.p.terminate()
        print("[CLIENT] Closed")


def main():
    client = CombinedClient()

    try:
        print("=" * 50)
        print("Combined TTS + ASR Test Client")
        print(f"ASR Server: {SERVER_ASR}")
        print(f"TTS Server: {SERVER_TTS}")
        print(f"Session ID: {client.session_id}")
        print("=" * 50)

        welcome = input("\nEnter welcome text (or press Enter for default): ").strip()
        if not welcome:
            welcome = "欢迎使用，请说话"

        duration_input = input("Recording duration in seconds (default 5): ").strip()
        try:
            duration = float(duration_input) if duration_input else 5.0
        except ValueError:
            duration = 5.0

        print(f"\n[MAIN] Welcome: {welcome}")
        print(f"[MAIN] Duration: {duration}s")
        print("=" * 50)

        result = client.speak_and_real_time_recognize(
            welcome_text=welcome,
            record_duration=duration,
        )

        print("\n" + "=" * 50)
        print("Final Results:")
        print(f"  ASR Results: {result['asr_results']}")
        print(f"  Recorded File: {result['recorded_file']}")
        print("=" * 50)

        again = input("\nTry again? (y/n): ").strip().lower()
        if again == "y":
            main()

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted")
    except Exception as e:
        print(f"[MAIN] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    main()
