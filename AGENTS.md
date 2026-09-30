# AGENTS.md - qwen3-asr-test

## Project Overview

This is a Python project for testing Qwen3-ASR (Automatic Speech Recognition) model. It uses real-time audio input from microphone and transcribes speech to text.

**Python Version:** 3.12  
**Package Manager:** uv  
**Key Dependencies:** pyaudio, qwen-asr, numpy, torch

## Build/Lint/Test Commands

### Package Management (uv)
```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Update lock file
uv lock

# Run with virtual environment
uv run python <script.py>
```

### Running the Application
```bash
# Run the main ASR test application
uv run python qwen_asr_test.py

# Run placeholder main
uv run python main.py
```

### Testing
```bash
# Run all tests (if any exist)
uv run pytest

# Run a single test file
uv run pytest tests/<test_file>.py

# Run a single test function
uv run pytest tests/<test_file>.py::test_function_name

# Run tests with verbose output
uv run pytest -v
```

### Linting/Type Checking
```bash
# Run linting (if configured)
uv run ruff check .

# Run ruff with auto-fix
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking (if mypy configured)
uv run mypy .
```

### Virtual Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Deactivate
deactivate
```

## Code Style Guidelines

### General
- Use Python 3.12+ features (type hints, pattern matching where appropriate)
- Keep functions focused and small (prefer single responsibility)
- Use meaningful variable names; avoid single letters except in short loops or lambdas

### Imports
```python
# Standard library imports first
import threading
import queue
import time

# Third-party imports second
import numpy as np
import torch

# Local/relative imports last
from qwen_asr import Qwen3ASRModel

# Sort imports alphabetically within each group
```

### Type Annotations
- Use type hints for function parameters and return values
- Use `typing.Optional` instead of `| None` for compatibility
- Use `typing.List`, `typing.Dict` etc. or modern `list[]`, `dict[]` syntax
```python
def process_audio(audio: np.ndarray, rate: int) -> Optional[str]:
    ...
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Constants | UPPER_SNAKE_CASE | `MAX_SAMPLES`, `CHUNK_SIZE` |
| Variables | snake_case | `audio_buf`, `stable_text` |
| Functions | snake_case | `def infer():`, `def process_audio():` |
| Classes | PascalCase | `MyClass` |
| Module files | snake_case | `qwen_asr_test.py` |

### Error Handling
```python
# Use specific exception types
try:
    result = queue.get_nowait()
except queue.Empty:
    pass  # Handle absence of result gracefully

# Always clean up resources in finally blocks
try:
    stream.start_stream()
except Exception:
    logging.error("Stream failed")
finally:
    stream.stop_stream()
    p.terminate()
```

### Threading Patterns
- Use `threading.Event` for signaling between threads
- Set daemon=True for worker threads that shouldn't block shutdown
- Use `queue.Queue` for thread-safe communication
```python
inference_done = threading.Event()
result_queue = queue.Queue()

# In worker thread
threading.Thread(target=infer, daemon=True).start()
```

### Audio Processing
- Use int16 for raw audio bytes, convert to float32 [-1, 1] for processing
- Normalize audio: `data.astype(np.float32) / 32768.0`
- Keep audio buffers bounded to prevent memory issues
```python
if len(audio_buf) > MAX_SAMPLES:
    audio_buf = audio_buf[-MAX_SAMPLES:]
```

### Logging/Debugging
- Use print with prefixes for debugging: `print(f"[DEBUG] message")`
- Use flush=True when print output needs to appear immediately
- Use `[INFO]`, `[DEBUG]`, `[ERROR]` prefixes consistently

### File Structure
```
qwen3_asr_test/
├── main.py              # Entry point placeholder
├── qwen_asr_test.py     # Main ASR application
├── pyproject.toml       # Project configuration
├── uv.lock             # Locked dependencies
├── .python-version     # Python version specification
├── models/             # Model files (Qwen3-ASR-0.6B)
└── AGENTS.md           # This file
```

### Performance Considerations
- Use `exception_on_overflow=False` for audio stream reading
- Batch inference to reduce overhead (process every N chunks)
- Consider memory constraints when buffering audio
- Use appropriate dtype for model (float16 for MPS, float32 for CPU)

## Common Patterns

### Model Loading
```python
model = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    device_map=DEVICE,
    max_new_tokens=256,
)
```

### Audio Transcription
```python
results = model.transcribe(
    audio=(audio, RATE),
    language=LANGUAGE,
)
text = results[0].text.strip() if results else ""
```

### Graceful Shutdown
```python
try:
    while True:
        # main loop
except KeyboardInterrupt:
    print("\n\n✅ 结束")
finally:
    # cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
```
