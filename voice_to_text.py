import pyaudio
import wave
import threading
import sys
import whisper
import pyperclip

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "recorded.wav"

# Flag to stop recording
recording = True

def record_audio():
    global recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Type 'e' or 'q' + Enter to stop.")
    frames = []

    while recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("Stopping recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def keyboard_listener():
    global recording
    while True:
        user_input = input()
        if user_input.strip().lower() in ['e', 'q']:
            recording = False
            break

if __name__ == "__main__":
    # Start keyboard listener in a separate thread
    kb_thread = threading.Thread(target=keyboard_listener)
    kb_thread.start()

    # Record audio in main thread
    record_audio()

    # Transcribe with Whisper
    print("Transcribing with Whisper...")
    model = whisper.load_model("turbo")  # or "small", "medium", "large" if you want better accuracy
    result = model.transcribe(WAVE_OUTPUT_FILENAME)
    text = result["text"].strip()
    print("\nTranscribed text:\n", text)

    # Copy to clipboard
    pyperclip.copy(text)
    print("\n✅ Transcription copied to clipboard!")
