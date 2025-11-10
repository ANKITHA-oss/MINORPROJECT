import sounddevice as sd
import soundfile as sf

def record_voice(filename, duration=5, samplerate=44100):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Recording saved as {filename}")

if __name__ == "__main__":
    record_voice("user_voice.wav")
