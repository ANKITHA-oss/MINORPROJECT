import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet

# -------------------------
# SECTION 1: Encryption Setup
# -------------------------
def load_key():
    return open("secret.key", "rb").read()

def encrypt_voice(filename):
    key = load_key()
    fernet = Fernet(key)
    with open(filename, "rb") as file:
        data = file.read()
    enc_data = fernet.encrypt(data)
    with open(filename + ".encrypted", "wb") as enc_file:
        enc_file.write(enc_data)
    print("🔒 Voice encrypted for privacy.")

# -------------------------
# SECTION 2: Voice Authentication
# -------------------------
def extract_features(filename):
    y, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def verify_voice(reference_file, test_file, threshold=0.8):
    ref = extract_features(reference_file).reshape(1, -1)
    test = extract_features(test_file).reshape(1, -1)
    sim = cosine_similarity(ref, test)[0][0]
    print(f"🧠 Voice similarity: {sim:.2f}")
    return sim >= threshold

# -------------------------
# SECTION 3: Keyword Detection
# -------------------------
def detect_keywords(text):
    keywords = ["help", "sos", "save me", "emergency"]
    for word in keywords:
        if word.lower() in text.lower():
            print("🚨 Emergency keyword detected:", word)
            return True
    return False

# -------------------------
# SECTION 4: Voice Recording + Recognition
# -------------------------
def record_and_recognize():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("🗣️ You said:", text)
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
        except sr.RequestError:
            print("⚠️ Network error.")
    return ""

# -------------------------
# MAIN SYSTEM
# -------------------------
if __name__ == "__main__":
    print("\n🔐 Voice & Speech Recognition — Security System Started 🔐\n")
    authorized = verify_voice("user_voice.wav", "user_voice.wav")  # simulate login

    if authorized:
        print("✅ Authorized user detected.")
        text = record_and_recognize()

        if detect_keywords(text):
            print("🚨 Sending secure alert to backend (placeholder)...")
            # Here you can later connect to your web API securely
        else:
            print("✅ No emergency detected.")
    else:
        print("❌ Unauthorized voice. Access denied.")

    encrypt_voice("user_voice.wav")
    print("🛡️ Privacy secured — session complete.")
