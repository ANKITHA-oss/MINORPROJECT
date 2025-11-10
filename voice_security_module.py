"""
voice_security_module.py

Provides:
 - initialize_models(vosk_model_path)
 - generate_key()  # run once to create secret.key
 - enroll_user_from_bytes(username, audio_bytes, fmt)
 - verify_user_from_bytes(audio_bytes, fmt, threshold=0.70)
 - transcribe_bytes(audio_bytes, fmt)
 - load_templates() / save_templates()  (internal)

Outputs are JSON-serializable dicts.

Privacy behavior:
 - temporary WAV files created and removed automatically
 - templates (speaker embeddings) stored encrypted on disk (voice_templates.json)
"""

import os
import io
import json
import base64
import tempfile
import shutil
from typing import Optional, Dict, Any, List

import numpy as np
from numpy.linalg import norm
from pydub import AudioSegment
import soundfile as sf

# ASR
from vosk import Model, KaldiRecognizer

# Speaker embeddings
from resemblyzer import VoiceEncoder, preprocess_wav

# Encryption
from cryptography.fernet import Fernet

# Paths / defaults
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # change if needed
TEMPLATES_FILE = "voice_templates.json"
KEY_FILE = "secret.key"

# Emergency/default phrases (backend may override)
DEFAULT_EMERGENCY_PHRASES = [
    "help", "help me", "i need help", "emergency", "sos", "save me",
    "call police", "call the police", "call 911", "get help"
]

# ---- Helpers: file/format conversions ----
def _to_wav_16k_mono_from_bytes(audio_bytes: bytes, input_format: str) -> str:
    """
    Convert incoming audio bytes (mp3/wav/m4a/ogg) to a temp 16k mono WAV file.
    Returns path to WAV file. Caller is responsible for deletion.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=input_format)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tf.close()
    out_path = tf.name
    audio.export(out_path, format="wav")
    return out_path

def _ensure_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at '{model_path}'. Download and set VOSK_MODEL_PATH.")
    return Model(model_path)

# ---- Encryption helpers ----
def generate_key():
    """Generate and save a Fernet key. Run once before using enroll/verify if secret.key missing."""
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"✅ Generated key and saved to {KEY_FILE}")

def _get_fernet():
    if not os.path.exists(KEY_FILE):
        raise FileNotFoundError(f"{KEY_FILE} not found. Run generate_key() first.")
    key = open(KEY_FILE, "rb").read()
    return Fernet(key)

def encrypt_bytes(b: bytes) -> str:
    f = _get_fernet()
    token = f.encrypt(b)
    return base64.b64encode(token).decode("utf-8")

def decrypt_bytes(b64: str) -> bytes:
    f = _get_fernet()
    token = base64.b64decode(b64.encode("utf-8"))
    return f.decrypt(token)

# ---- Templates storage ----
def save_templates(templates: Dict[str, str]):
    """templates: mapping username -> base64(encrypted bytes)"""
    with open(TEMPLATES_FILE, "w") as f:
        json.dump(templates, f)

def load_templates() -> Dict[str, str]:
    if not os.path.exists(TEMPLATES_FILE):
        return {}
    with open(TEMPLATES_FILE, "r") as f:
        return json.load(f)

# ---- similarity ----
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

# ---- Module class (bundles models) ----
class VoiceSecurityModule:
    def __init__(self, vosk_model_path: str = VOSK_MODEL_PATH):
        # Initialize Vosk model and VoiceEncoder lazily (may be heavy)
        self.vosk_model_path = vosk_model_path
        self.vosk_model = None
        self.voice_encoder = None

    def _init_vosk(self):
        if self.vosk_model is None:
            self.vosk_model = _ensure_model(self.vosk_model_path)

    def _init_encoder(self):
        if self.voice_encoder is None:
            self.voice_encoder = VoiceEncoder()

    # ---- Transcription ----
    def transcribe_wav_file(self, wav_path: str) -> Dict[str, Any]:
        self._init_vosk()
        results = []
        with open(wav_path, "rb") as wf:
            rec = KaldiRecognizer(self.vosk_model, 16000.0)
            rec.SetWords(True)
            while True:
                data = wf.read(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    j = json.loads(rec.Result())
                    results.append(j)
            j = json.loads(rec.FinalResult())
            results.append(j)
        transcript_parts = [r.get("text", "") for r in results if r.get("text")]
        transcript = " ".join(transcript_parts).strip()
        return {"transcript": transcript, "raw_results": results}

    def transcribe_bytes(self, audio_bytes: bytes, fmt: str) -> Dict[str, Any]:
        tmp = None
        try:
            tmp = _to_wav_16k_mono_from_bytes(audio_bytes, fmt)
            return self.transcribe_wav_file(tmp)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    # ---- Speaker enrollment/verification ----
    def enroll_user_from_bytes(self, username: str, audio_bytes: bytes, fmt: str) -> Dict[str, Any]:
        """
        Enroll a user by computing an embedding from provided audio bytes and saving it encrypted.
        Returns dict with success and message.
        """
        self._init_encoder()
        tmp = None
        try:
            tmp = _to_wav_16k_mono_from_bytes(audio_bytes, fmt)
            # Resemblyzer's preprocess_wav expects a path
            wav = preprocess_wav(tmp)
            emb = self.voice_encoder.embed_utterance(wav)  # 256-d float32
            # store raw bytes of the embedding (float32)
            emb_bytes = emb.tobytes()
            enc_b64 = encrypt_bytes(emb_bytes)
            templates = load_templates()
            templates[username] = enc_b64
            save_templates(templates)
            return {"ok": True, "msg": f"Enrolled {username} (template stored encrypted)."}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def verify_user_from_bytes(self, audio_bytes: bytes, fmt: str, threshold: float = 0.70) -> Dict[str, Any]:
        """
        Verify the incoming audio against all stored templates.
        Returns best match, score, and whether it passed the threshold.
        """
        self._init_encoder()
        tmp = None
        try:
            tmp = _to_wav_16k_mono_from_bytes(audio_bytes, fmt)
            wav = preprocess_wav(tmp)
            test_emb = self.voice_encoder.embed_utterance(wav)
            templates = load_templates()
            best_user = None
            best_score = -1.0
            for user, enc_b64 in templates.items():
                try:
                    emb_bytes = decrypt_bytes(enc_b64)
                    enrolled_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    score = cosine_similarity(test_emb, enrolled_emb)
                    if score > best_score:
                        best_score = score
                        best_user = user
                except Exception:
                    continue
            passed = best_score >= threshold if best_score >= 0 else False
            return {"ok": True, "best_user": best_user, "score": float(best_score), "passed": bool(passed)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    # ---- Combined convenience ----
    def process_audio_and_return_result(
        self,
        audio_bytes: bytes,
        fmt: str,
        emergency_phrases: Optional[List[str]] = None,
        verify_threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Full pipeline: transcribe -> detect emergency phrases -> verify speaker (if templates exist)
        Returns: { transcript, emergency: bool, matched_phrases, speaker: {best_user, score, passed} or None }
        """
        if emergency_phrases is None:
            emergency_phrases = DEFAULT_EMERGENCY_PHRASES

        # 1) transcribe
        trans = self.transcribe_bytes(audio_bytes, fmt)
        transcript = trans.get("transcript", "")

        matched = []
        lower = transcript.lower()
        for p in emergency_phrases:
            if p in lower:
                matched.append(p)
                matched = []
                # 🚨 If emergency detected, trigger backend alert
        if matched:
            import requests
            try:
                print("🚨 Emergency phrase detected — sending SOS alert...")
                latitude = 12.9716
                longitude = 77.5946
                
                response = requests.post(
                    "http://127.0.0.1:5000/sos",
                    json={"latitude": latitude, "longitude": longitude, "message": transcript}
                )

                if response.status_code == 200:
                    print("✅ SOS alert sent successfully!")
                else:
                    print(f"⚠️ Failed to send SOS alert: {response.text}")

            except Exception as e:
                print(f"❌ Error sending SOS alert: {e}")

        # 2) speaker verify if templates exist
        templates = load_templates()
        speaker_info = None
        if templates:
            speaker_info = self.verify_user_from_bytes(audio_bytes, fmt, threshold=verify_threshold)

        return {
            "transcript": transcript,
            "emergency": len(matched) > 0,
            "matched_phrases": matched,
            "speaker": speaker_info,
            "raw_asr": trans.get("raw_results", []),
        }
