import sys
import json
import pyaudio
import time
from vosk import Model, KaldiRecognizer
import subprocess

# ---- 1) SETUP ----

# Load your Vosk model once (small model recommended for real-time use)
print("Loading Voice Models...")
modellight = Model("stt-models/vosk-model-small-en-us-0.15")
modelheavy = Model("stt-models/vosk-model-en-us-0.22")
# First recognizer: wake word detection mode
wake_words = '["hello"]'  # your custom wake word
rec_wake = KaldiRecognizer(modellight, 16000, wake_words)

# Later we'll switch to full recognizer (free vocabulary)
rec_stt = KaldiRecognizer(modelheavy, 16000)

# Setup microphone stream
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=8000,
)
stream.start_stream()

print("✅ Listening for wake word:", wake_words)
while True:
    # ---- 2) MAIN LOOP ----
    while True:
        data = stream.read(4000, exception_on_overflow=False)

        # Wake word mode
        if rec_wake.AcceptWaveform(data):
            result = json.loads(rec_wake.Result())
            text = result.get("text", "")
            if "hello" in text:
                print("\n🚀 Wake word detected! Switching to full STT mode...")
                break



    # ---- 3) STT MODE ----
    print("🎙️  Speak now... (Ctrl+C to exit)")

    last_partial = ""
    start_time = time.time()

    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)

            if rec_stt.AcceptWaveform(data):
                # Final result — print on new line
                final = json.loads(rec_stt.Result()).get("text", "").strip()
                if final:
                    print("\r✅ Final:  ", final, "")
                    last_partial = ""
                    start_time = time.time()
            else:
                # Partial result — update same line
                partial = json.loads(rec_stt.PartialResult()).get("partial", "").strip()
                if partial != last_partial:
                    sys.stdout.write("\r📝 Partial:" + partial + " " * 20)
                    sys.stdout.flush()
                    last_partial = partial

            # Optional: exit after 10 seconds of inactivity
            if time.time() - start_time > 10:
                print("\n⏰ Timeout: returning to wake word mode.")
                break

    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        sys.exit(0)
