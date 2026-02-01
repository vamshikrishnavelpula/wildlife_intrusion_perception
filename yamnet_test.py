import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf
import csv
import os

# === Paths ===
MODEL_PATH = "/home/user/wildlife/yamnet/archive"  # SavedModel folder
CLASS_MAP_PATH = "/home/user/wildlife/yamnet/yamnet_class_map.csv"
SOUND_FILE = "/home/user/wildlife/test_sound.wav"

# === Load model ===
print("Loading YAMNet model...")
model = tf.saved_model.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# === Load class map ===
class_names = []
with open(CLASS_MAP_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row["display_name"])

# === Load and preprocess audio ===
print(f"Reading audio file: {SOUND_FILE}")
waveform, sr = sf.read(SOUND_FILE)
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)  # Convert to mono
if sr != 16000:
    print(f"Resampling from {sr}Hz to 16000Hz...")
    import librosa
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    sr = 16000

waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

# === Run inference ===
print("Running inference...")
scores, embeddings, spectrogram = model(waveform)
scores_np = scores.numpy().mean(axis=0)
top5_i = np.argsort(scores_np)[::-1][:5]

print("\n🔊 Top 5 Predictions:")
for i in top5_i:
    print(f"• {class_names[i]} ({scores_np[i]*100:.2f}%)")

