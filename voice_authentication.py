import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from sklearn.metrics.pairwise import cosine_similarity

# --- Function to record new voice sample ---
def record_new_voice(filename="new_voice.wav", duration=4, samplerate=44100):
    print("🎙️ Please speak for a few seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("✅ New voice recorded.")

# --- Function to extract MFCC features ---
def extract_features(filename):
    y, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# --- Function to compare two voices ---
def verify_voice(reference_file, test_file, threshold=0.8):
    ref_features = extract_features(reference_file).reshape(1, -1)
    test_features = extract_features(test_file).reshape(1, -1)

    similarity = cosine_similarity(ref_features, test_features)[0][0]
    print(f"🧠 Voice similarity: {similarity:.2f}")

    if similarity >= threshold:
        print("✅ Access Granted! Voice verified successfully.")
    else:
        print("❌ Access Denied! Voice does not match.")

# --- Main execution ---
if __name__ == "__main__":
    record_new_voice()
    verify_voice("user_voice.wav", "new_voice.wav")
