import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Use microphone as source
with sr.Microphone() as source:
    print("🎙️ Speak something... (I’m listening)")
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("✅ You said:", text)
    except sr.UnknownValueError:
        print("❌ Sorry, I could not understand your voice.")
    except sr.RequestError:
        print("⚠️ Network error. Please check your internet connection.")
