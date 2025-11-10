import os
import torch
import speech_recognition as sr
import pyttsx3
from cryptography.fernet import Fernet
import torchaudio
from silero_vad import get_speech_timestamps
import tkinter as tk
from tkinter import messagebox

# --------------------------
# 1️⃣ Security Utilities
# --------------------------

def generate_key():
    if not os.path.exists("secret.key"):
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)
        print("✅ Key generated.")
    else:
        print("🔑 Key already exists.")

def load_key():
    return open("secret.key", "rb").read()

def encrypt_file(file_path):
    key = load_key()
    fernet = Fernet(key)
    with open(file_path, "rb") as file:
        data = file.read()
    encrypted = fernet.encrypt(data)
    with open(file_path + ".encrypted", "wb") as enc_file:
        enc_file.write(encrypted)
    print(f"🔒 Encrypted: {file_path}")

def decrypt_file(encrypted_path):
    key = load_key()
    fernet = Fernet(key)
    with open(encrypted_path, "rb") as enc_file:
        data = enc_file.read()
    decrypted = fernet.decrypt(data)
    original_path = encrypted_path.replace(".encrypted", "")
    with open(original_path, "wb") as dec_file:
        dec_file.write(decrypted)
    print(f"🔓 Decrypted: {encrypted_path}")

# --------------------------
# 2️⃣ Speech & Recognition
# --------------------------

def recognize_speech():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    status_label.config(text="🎙️ Listening...")
    root.update()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        with open("user_voice.wav", "wb") as f:
            f.write(audio.get_wav_data())

    status_label.config(text="🎧 Recognizing speech...")
    root.update()

    try:
        text = recognizer.recognize_google(audio)
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, text)
        print(f"You said: {text}")

        # Save and encrypt
        with open("recognized_text.txt", "w") as file:
            file.write(text)

        encrypt_file("recognized_text.txt")
        encrypt_file("user_voice.wav")

        # Check emergency words
        if any(word in text.lower() for word in ["help", "sos", "emergency", "save me"]):
            messagebox.showwarning("🚨 Alert!", "Emergency keyword detected!")
            engine.say("Alert triggered. Help message sent securely.")
            engine.runAndWait()
        else:
            messagebox.showinfo("✅ Done", "No emergency words detected.")
            engine.say("Processing complete.")
            engine.runAndWait()

        status_label.config(text="✅ Process complete.")

    except sr.UnknownValueError:
        messagebox.showerror("Error", "Could not understand the voice.")
        status_label.config(text="❌ Try again.")
    except sr.RequestError:
        messagebox.showerror("Error", "Speech recognition service unavailable.")
        status_label.config(text="⚠️ Network issue.")

# --------------------------
# 3️⃣ GUI Setup
# --------------------------

root = tk.Tk()
root.title("🎤 Voice & Text Security System")
root.geometry("500x400")
root.config(bg="#1e1e1e")

title_label = tk.Label(root, text="Voice & Text Recognition - Secure System", fg="white", bg="#1e1e1e", font=("Arial", 14, "bold"))
title_label.pack(pady=15)

text_box = tk.Text(root, height=6, width=50, wrap=tk.WORD, bg="#2b2b2b", fg="white", font=("Arial", 11))
text_box.pack(pady=10)

record_btn = tk.Button(root, text="🎙️ Start Recording", font=("Arial", 12, "bold"), bg="#0078D7", fg="white", command=recognize_speech)
record_btn.pack(pady=8)

decrypt_btn = tk.Button(root, text="🔓 Decrypt Files", font=("Arial", 12, "bold"), bg="#3AA655", fg="white",
                        command=lambda: [decrypt_file("recognized_text.txt.encrypted"), decrypt_file("user_voice.wav.encrypted")])
decrypt_btn.pack(pady=8)

status_label = tk.Label(root, text="Ready to record 🎧", fg="lightgray", bg="#1e1e1e", font=("Arial", 10))
status_label.pack(pady=20)

generate_key()
root.mainloop()
