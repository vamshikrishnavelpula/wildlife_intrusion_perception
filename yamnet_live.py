import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import time, datetime, os
import csv

# --- CONFIG ---
MODEL_PATH = "/home/user/wildlife/yamnet/archive"
CSV_PATH = "/home/user/wildlife/yamnet/yamnet_class_map.csv"
LOG_FILE = "sound_detections.log"
SAMPLE_RATE = 16000
DURATION = 5  # seconds per sample

# --- Load YAMNet model ---
print("🔄 Loading YAMNet model...")
model = hub.load(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# --- Load class labels properly ---
class_names = []
with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'].strip().lower())

# --- Target classes ---
target_map = {
    "elephant": ["elephant", "elephant trumpeting", "trumpeting", "roar","engine","vehicle","siren","air horn","truck horn"],
    "boar": ["wild boar", "pig", "oink", "grunt","frog","wild animals","croak","squeal"],
    "monkey": ["monkey", "chimpanzee", "howler monkey", "gibbon", "crying", "screaming","sobbing","owl"]
}

# --- Reverse lookup for detection ---
reverse_map = {label: key for key, group in target_map.items() for label in group}

# --- Fallback keywords ---
fallback_keywords = ["animal", "wild animals", "domestic animals", "roaring cats"]

# --- Prepare log file ---
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,animal,confidence\n")

print("🎧 Listening for ELEPHANT / BOAR / MONKEY sounds... (Ctrl+C to stop)\n")

try:
    while True:
        # Record audio
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        waveform = np.squeeze(audio)

        # Run model inference
        scores, embeddings, spectrogram = model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        # Get top 5 predictions
        top5 = np.argsort(mean_scores)[::-1][:5]

        # Display top 5 with readable names
        top_predictions = [(class_names[i].capitalize(), float(mean_scores[i]*100)) for i in top5]
        print("\n🎯 Top predictions:", top_predictions)

        detected = False

        # Check for target animal sounds
        for idx in top5:
            label = class_names[idx]
            conf = mean_scores[idx]

            if label in reverse_map and conf > 0.15:
                animal = reverse_map[label]
                detected = True
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"🐾 [{timestamp}] DETECTED: {animal.upper()} ({conf*100:.1f}%)")

                with open(LOG_FILE, "a") as f:
                    f.write(f"{timestamp},{animal},{conf:.4f}\n")
                break

        # Fallback for general animal sounds
        if not detected:
            for idx in top5:
                label = class_names[idx]
                conf = mean_scores[idx]
                if label in fallback_keywords and conf > 0.2:
                    detected = True
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"🐾 [{timestamp}] DETECTED: GENERAL ANIMAL ({conf*100:.1f}%)")
                    with open(LOG_FILE, "a") as f:
                        f.write(f"{timestamp},general_animal,{conf:.4f}\n")
                    break

        # No detection
        if not detected:
            print(".", end="", flush=True)

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    print(f"📁 Logs saved to: {os.path.abspath(LOG_FILE)}")

