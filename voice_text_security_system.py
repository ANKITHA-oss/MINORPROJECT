import os
import torch
import speech_recognition as sr
import pyttsx3
from cryptography.fernet import Fernet
import torchaudio
from silero_vad import get_speech_timestamps, collect_chunks
from datetime import datetime

# --------------------------
# 1️⃣ Encryption utilities
# --------------------------

def generate_key():
    """Generate a key for encrypting audio/text data."""
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
    print("✅ Encryption key generated: secret.key")

def load_key():
    return open("secret.key", "rb").read()

def encrypt_file(file_path):
    """Encrypts a file (audio or text)."""
    key = load_key()
    fernet = Fernet(key)
    with open(file_path, "rb") as file:
        data = file.read()
    encrypted = fernet.encrypt(data)
    with open(file_path + ".encrypted", "wb") as enc_file:
        enc_file.write(encrypted)
    print(f"🔒 Encrypted: {file_path} -> {file_path}.encrypted")

def decrypt_file(encrypted_path):
    key = load_key()
    fernet = Fernet(key)
    with open(encrypted_path, "rb") as enc_file:
        data = enc_file.read()
    decrypted = fernet.decrypt(data)
    original_path = encrypted_path.replace(".encrypted", "")
    with open(original_path, "wb") as dec_file:
        dec_file.write(decrypted)
    print(f"🔓 Decrypted: {encrypted_path} -> {original_path}")

# --------------------------
# 2️⃣ Voice Activity Detection
# --------------------------

def detect_speech_segments(audio_path):
    wav, sr = torchaudio.load(audio_path)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _) = utils
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
    print(f"🗣️ Detected {len(speech_timestamps)} speech segments.")
    return speech_timestamps

# --------------------------
# 3️⃣ Speech-to-Text & Alerts
# --------------------------

def recognize_and_secure():
    r = sr.Recognizer()
    engine = pyttsx3.init()
    print("🎤 Speak now...")

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        with open("user_voice.wav", "wb") as f:
            f.write(audio.get_wav_data())

    print("🎧 Recognizing speech...")
    try:
        text = r.recognize_google(audio)
        print(f"🗣️ You said: {text}")
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return
    except sr.RequestError:
        print("⚠️ Speech Recognition API unavailable.")
        return

    # Save recognized text
    with open("recognized_text.txt", "w") as file:
        file.write(text)

    # Encrypt both text and audio
    encrypt_file("recognized_text.txt")
    encrypt_file("user_voice.wav")

    # Check for emergency keywords
    if any(word in text.lower() for word in ["help", "sos", "emergency", "save me"]):
        print("🚨 Emergency keyword detected!")
        engine.say("Alert triggered. Stay calm, we are sending help.")
        engine.runAndWait()
        # Placeholder for backend alert:
        # send_alert_to_backend(text)
    else:
        print("✅ No emergency keywords found.")

# --------------------------
# 4️⃣ Run module
# --------------------------

if __name__ == "__main__":
    if not os.path.exists("secret.key"):
        generate_key()
    recognize_and_secure()
