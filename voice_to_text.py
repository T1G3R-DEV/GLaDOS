import pyaudio
import numpy as np
import whisper
import pyperclip
import webrtcvad
import io
import wave
import threading
import soundfile as sf  # NEW: to read in-memory WAV into numpy

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # 1024 samples = ~64ms at 16kHz
VAD_FRAME_MS = 30  # webrtcvad frame duration (10, 20, or 30 ms)

vad = webrtcvad.Vad(2)  # aggressiveness mode 0-3 (higher = more aggressive silence detection)
recording = True
final_text = []

def keyboard_listener():
    global recording
    while True:
        user_input = input()
        if user_input.strip().lower() in ['e', 'q']:
            recording = False
            break

def record_and_transcribe():
    global recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Listening with VAD... Speak! Type 'e' or 'q' + Enter to stop.\n")

    model = whisper.load_model("small")  # local whisper model

    buffer = []
    silence_duration = 0
    max_silence_ms = 800  # how long silence must last before ending a segment

    while recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        buffer.append(data)

        # split chunk into VAD frames
        frame_bytes = int(RATE * (VAD_FRAME_MS / 1000.0) * 2)  # samples * 2 bytes/sample
        for i in range(0, len(data), frame_bytes):
            frame = data[i:i + frame_bytes]
            if len(frame) < frame_bytes:
                continue  # skip incomplete frame at end of chunk

            is_speech = vad.is_speech(frame, RATE)

            if is_speech:
                silence_duration = 0
            else:
                silence_duration += VAD_FRAME_MS

            if silence_duration > max_silence_ms:
                # end of phrase detected
                if len(buffer) > 0:
                    #print("\n[Detected end of speech segment]")

                    # concatenate audio and write to in-memory WAV
                    audio_data = b''.join(buffer)
                    wav_bytes = io.BytesIO()
                    wf = wave.open(wav_bytes, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(audio_data)
                    wf.close()
                    wav_bytes.seek(0)

                    # 🔥 read WAV in RAM into numpy
                    audio_np, sr = sf.read(wav_bytes, dtype="float32")
                    if sr != RATE:
                        print(f"Warning: sample rate mismatch (expected {RATE}, got {sr})")

                    # Transcribe numpy array
                    result = model.transcribe(audio_np, fp16=False)
                    text = result["text"].strip()
                    if text:
                        print(f"> {text}")
                        final_text.append(text)

                buffer = []
                silence_duration = 0  # reset silence tracker

    # Transcribe anything left in the buffer on exit
    if buffer:
        print("\n[Finalizing last segment]")
        audio_data = b''.join(buffer)
        wav_bytes = io.BytesIO()
        wf = wave.open(wav_bytes, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
        wf.close()
        wav_bytes.seek(0)

        audio_np, sr = sf.read(wav_bytes, dtype="float32")
        if sr != RATE:
            print(f"Warning: sample rate mismatch (expected {RATE}, got {sr})")

        result = model.transcribe(audio_np, fp16=False)
        text = result["text"].strip()
        if text:
            print(f"> {text}")
            final_text.append(text)

    print("Stopping recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    kb_thread = threading.Thread(target=keyboard_listener)
    kb_thread.start()

    record_and_transcribe()

    transcript = "\n".join(final_text)
    print("\nFinal transcription:\n", transcript)
    pyperclip.copy(transcript)
    print("\n✅ Transcription copied to clipboard!")
