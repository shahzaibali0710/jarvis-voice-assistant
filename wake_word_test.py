"""
Jarvis - Step 1: Wake Word Detection Test
Listens for the wake word "hey jarvis" using openWakeWord (fully local, no API key).
Just proves the mic -> wake word pipeline works. No LLM/TTS yet.
"""

import pyaudio
import numpy as np
from openwakeword.model import Model

# Load the pre-trained "hey jarvis" model (bundled with openwakeword)
owwModel = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

# Audio config required by openWakeWord
CHUNK = 1280  # 80ms at 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print("Listening for wake word 'Hey Jarvis'... (press Ctrl+C to stop)")

try:
    while True:
        # Read audio chunk from mic
        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

        # Run prediction
        prediction = owwModel.predict(audio_data)

        # Check score for our wake word
        for wakeword, score in prediction.items():
            if score > 0.5:
                print(f"Wake word detected! ({wakeword}, score={score:.2f})")

except KeyboardInterrupt:
    print("\nStopped listening.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()