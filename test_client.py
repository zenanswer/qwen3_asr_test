import io
import os
import queue
import threading
import time
import wave

import numpy as np
import pyaudio
import requests
import soundfile as sf

SERVER_URL = "http://10.10.11.32:8001"
RATE = 16000
CHUNK = 1600
TTS_SAMPLE_RATE = 24000


class CombinedClient:
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        self.session_id = int(time.time() * 1000) % 1000000
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_streaming = False

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
            self.is_streaming = True

    def _close_stream(self):
        if self.stream is not None:
            self.is_streaming = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def speak_and_real_time_recognize(
        self,
        text: str,
        voice_description: str = "男声，语速中等",
        language: str = "auto",
        record_duration: float = 5.0,
    ) -> dict:
        """
        1. 调用 TTS 生成语音并播放
        2. 同时开始录音
        3. 返回识别结果

        Args:
            text: 要说话的文本
            voice_description: 声音描述
            language: 语言
            record_duration: 录音时长（秒）

        Returns:
            dict: {"tts_audio": bytes, "asr_results": list, "recorded_file": str}
        """
        print(f"[CLIENT] Generating TTS for: {text[:50]}...")
        tts_audio = self._call_tts(text, voice_description, language)

        print(f"[CLIENT] Playing TTS audio and recording from mic...")
        recorded_audio = self._play_and_record(tts_audio, record_duration)

        print(f"[CLIENT] Sending to ASR...")
        asr_results = self._send_to_asr_and_get_results(recorded_audio)

        return {
            "tts_audio": tts_audio,
            "asr_results": asr_results,
        }

    def _call_tts(self, text: str, voice_description: str, language: str) -> bytes:
        response = requests.post(
            f"{self.server_url}/tts",
            json={
                "text": text,
                "voice_description": voice_description,
                "language": language,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(f"TTS failed: {response.status_code} - {response.text}")
        return response.content

    def _play_and_record(
        self, audio_bytes: bytes, record_duration: float
    ) -> np.ndarray:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if sr != RATE:
            import scipy.signal as scipy_signal

            num_samples = int(len(audio_data) * RATE / sr)
            audio_data = scipy_signal.resample(audio_data, num_samples)
            sr = RATE

        audio_int16 = (audio_data * 32768).astype(np.int16)

        record_frames = int(RATE / CHUNK * record_duration)
        recorded_frames = []

        self._init_stream()

        def play_audio():
            try:
                self.stream.write(audio_int16.tobytes())
            except Exception as e:
                print(f"[CLIENT] Playback error: {e}")

        play_thread = threading.Thread(target=play_audio)
        play_thread.start()

        print(f"[CLIENT] Recording for {record_duration}s...")
        for _ in range(record_frames):
            if not self.is_streaming:
                break
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                recorded_frames.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                print(f"[CLIENT] Recording error: {e}")
                break

        play_thread.join(timeout=1)

        if recorded_frames:
            recorded_np = np.concatenate(recorded_frames)
        else:
            recorded_np = np.array([], dtype=np.int16)

        self._close_stream()

        timestamp = int(time.time() * 1000)
        record_path = f"./test_recordings/{self.session_id}_{timestamp}.wav"
        with wave.open(record_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(recorded_np.tobytes())
        print(f"[CLIENT] Saved recording to {record_path}")

        return recorded_np

    def _send_to_asr_and_get_results(self, audio: np.ndarray) -> list:
        audio_float32 = audio.astype(np.float32) / 32768.0

        response = requests.post(
            f"{self.server_url}/asr/audio/{self.session_id}",
            data=audio.tobytes(),
            timeout=30,
        )

        if response.status_code != 200:
            print(f"[CLIENT] ASR send error: {response.status_code}")
            return []

        result = response.json()
        results = result.get("results", [])

        for i, text in enumerate(results):
            print(f"[CLIENT] ASR Result {i + 1}: {text}")

        return results

    def get_asr_results(self) -> list:
        response = requests.get(
            f"{self.server_url}/asr/results/{self.session_id}",
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
        print("Combined ASR + TTS Test Client")
        print(f"Server: {SERVER_URL}")
        print(f"Session ID: {client.session_id}")
        print("=" * 50)

        test_text = input(
            "\nEnter text to speak (or press Enter for default): "
        ).strip()
        if not test_text:
            test_text = "你好，这是一个测试。请问今天天气怎么样？"

        print(f"\n[MAIN] Text: {test_text}")

        result = client.speak_and_real_time_recognize(
            text=test_text,
            voice_description="男声，语速中等",
            language="auto",
            record_duration=5.0,
        )

        print("\n" + "=" * 50)
        print("Results:")
        print(f"  ASR Results: {result['asr_results']}")
        print("=" * 50)

        another = input("\nTry again? (y/n): ").strip().lower()
        if another == "y":
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
