import torch
import torchaudio
from silero_vad import get_speech_timestamps, collect_chunks

def detect_speech_segments(audio_path):
    # load audio
    wav, sr = torchaudio.load(audio_path)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    get_speech_timestamps, _, _, _ = utils
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
    print("Speech segments detected:", speech_timestamps)
    return speech_timestamps
