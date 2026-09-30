import io
import logging
import os
import sys
import threading
import time
import wave

import numpy as np
import pyaudio
import requests
import scipy.signal as scipy_signal
import soundfile as sf
import websockets
import json
import asyncio


class UnbufferedStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)

    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[UnbufferedStreamHandler()],
)
logger = logging.getLogger(__name__)

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
        self.output_stream = None
        self.is_streaming = False
        self.all_results = []
        self.first_result_time = None
        self.last_result_time = None

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

    def _init_output_stream(self):
        if self.output_stream is None:
            self.output_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK,
            )

    def _close_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.output_stream is not None:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None

    def speak_and_real_time_recognize_audio(
        self,
        text: str,
        first_idle_timeout: float = 10.0,
        idle_timeout: float = 3.0,
        total_timeout: float = 30.0,
        expected_phrase: str = None,
    ) -> str:
        self.expected_phrase = (
            expected_phrase.lower().strip() if expected_phrase else None
        )
        logger.info(f"Generating TTS for: {text[:50]}...")
        tts_audio = self._call_tts(text)

        logger.info("Playing TTS...")
        self._play_tts_audio(tts_audio)
        logger.info("TTS finished, starting ASR...")

        self.all_results = []
        self.first_result_time = None
        self.last_result_time = None
        self.is_streaming = True

        self.first_idle_timeout = first_idle_timeout
        self.idle_timeout = idle_timeout
        self.total_timeout = total_timeout

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_asr_session())
        except Exception as e:
            logger.error(f"ASR error: {e}")

        logger.info(f"Returning results: {self.all_results}")
        if self.expected_phrase and any(
            self.expected_phrase in r.lower() for r in self.all_results
        ):
            logger.info("Expected phrase found, exiting directly")
            sys.exit(0)
        return " ".join(self.all_results)

    def _call_tts(self, text: str) -> bytes:
        response = requests.post(
            f"{self.tts_url}/tts",
            json={
                "text": text,
                "voice_description": "女声，语速中等",
                "language": "auto",
            },
            timeout=120,
        )
        if response.status_code != 200:
            raise Exception(f"TTS failed: {response.status_code} - {response.text}")
        return response.content

    def _play_tts_audio(self, audio_bytes: bytes):
        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if sr != RATE:
            num_samples = int(len(audio_data) * RATE / sr)
            audio_data = scipy_signal.resample(audio_data, num_samples)

        audio_int16 = (audio_data * 32768).astype(np.int16)

        self._init_output_stream()
        try:
            self.output_stream.write(audio_int16.tobytes())
        except Exception as e:
            logger.error(f"Playback error: {e}")

    async def _run_asr_session(self):
        host = self.asr_url.replace("http://", "")
        uri = f"ws://{host}/ws/asr/{self.session_id}"
        logger.info(f"Connecting to {uri}")

        async with websockets.connect(uri, ping_timeout=None) as websocket:
            recv_task = asyncio.create_task(self._recv_messages(websocket))
            await self._send_audio_loop(websocket)
            await asyncio.sleep(0.1)
            self.is_streaming = False
            try:
                await asyncio.wait_for(recv_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    async def _recv_messages(self, websocket):
        consecutive_timeouts = 0
        while True:
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                consecutive_timeouts = 0
                data = json.loads(msg)

                if data.get("type") == "result":
                    text = data.get("text", "")
                    if text and text not in self.all_results:
                        self.all_results.append(text)
                        now = time.time()
                        if self.first_result_time is None:
                            self.first_result_time = now
                        self.last_result_time = now
                        logger.info(f"SENTENCE: {text}")
                        if (
                            self.expected_phrase
                            and self.expected_phrase in text.lower()
                        ):
                            logger.info(
                                f"Found expected phrase: {self.expected_phrase}"
                            )
                            self.expected_phrase_found = True
                            await websocket.close()
                            return

            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                if not self.is_streaming:
                    break
                if consecutive_timeouts >= 3:
                    break
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"recv error: {e}")
                break
            except Exception as e:
                logger.error(f"recv error: {e}")
                break

    async def _send_audio_loop(self, websocket):
        self._init_stream()

        start_time = time.time()
        send_count = 0
        loop = asyncio.get_event_loop()

        while self.is_streaming:
            elapsed = time.time() - start_time

            if elapsed > self.total_timeout:
                logger.warning(f"total_timeout: {elapsed:.1f}s > {self.total_timeout}s")
                break

            if self.first_result_time and self.last_result_time:
                idle = time.time() - self.last_result_time
                if idle > self.idle_timeout:
                    logger.warning(f"idle_timeout: {idle:.1f}s > {self.idle_timeout}s")
                    break

            if not self.first_result_time and elapsed > self.first_idle_timeout:
                logger.warning(
                    f"first_idle_timeout: {elapsed:.1f}s > {self.first_idle_timeout}s"
                )
                break

            try:
                chunk = await loop.run_in_executor(None, self.stream.read, CHUNK)
            except Exception as e:
                logger.error(f"Read error: {e}")
                break

            try:
                await websocket.send(chunk)
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

            send_count += 1

    def close(self):
        self.is_streaming = False
        self._close_stream()
        self.p.terminate()
        logger.info("Closed")


def main():
    client = CombinedClient()

    try:
        logger.info("=" * 50)
        logger.info("Combined TTS + ASR Test Client")
        logger.info(f"ASR Server: {SERVER_ASR}")
        logger.info(f"TTS Server: {SERVER_TTS}")
        logger.info(f"Session ID: {client.session_id}")
        logger.info("=" * 50)

        text = input("\nEnter text to speak: ").strip()
        if not text:
            text = "欢迎使用，请说话"

        logger.info(f"Text: {text}")
        logger.info("=" * 50)

        result = client.speak_and_real_time_recognize_audio(
            text=text,
            first_idle_timeout=10.0,
            idle_timeout=3.0,
            total_timeout=30.0,
            expected_phrase="退下吧",
        )

        logger.info("=" * 50)
        logger.info(f"Final Result: {result}")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.warning("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    main()
