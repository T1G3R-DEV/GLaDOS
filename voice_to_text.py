import pyaudio
from vosk import Model, KaldiRecognizer

# 1) Load the model
model = Model("./stt-models/vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)

# 2) Setup microphone input with PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Listening...")

while True:
    data = stream.read(4000, exception_on_overflow=False)
    if rec.AcceptWaveform(data):
        print(rec.Result())   # prints JSON with 'text' key
    else:
        print(rec.PartialResult())  # partial result as you speak
