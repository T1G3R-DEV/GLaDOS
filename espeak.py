import subprocess

def speak(text):
    """
    Uses espeak-ng to synthesize speech.
    Adjust -s (speed), -p (pitch), -v (voice) to taste.
    """
    cmd = [
        "espeak-ng",
        "-s", "200",       # speed (words per minute)
        "-p", "90",        # pitch (lower = more robotic)
        "-v", "en-us",     # voice/language
        text
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    text = "Hello. I am your robotic assistant. Testing one two three."
    print(f"Speaking: {text}")
    speak(text)
