⛳️YAMNet Algorithm

Algorithm Overview :

YAMNet is a deep audio event classifier based on MobileNetV1, trained on the AudioSet dataset (over 500+ sound classes) by Google.
It identifies sound events using mel-spectrogram features.

Algorithm Steps:

1)Audio Input:
Capture a sound waveform at 16 kHz(16,000 readings per second) keeps the audio quality clear enough for speech and animal sounds.
->trained at 16 kHz

2)Feature Extraction:
Convert raw waveform → log mel-spectrogram using short-time Fourier transform (STFT).
->A spectrogram is a 2D image showing how sound frequencies vary over time.
    X-axis → time
    Y-axis → frequency
    Colors → loudness/intensity

3)Embedding Extraction:
Pass spectrogram through MobileNet layers (CNN used for image recognition) to get embeddings.
-->MobileNet treats spectrogram as an image.
	It extracts key patterns like:
	frequency changes (pitch),
	texture of sound(roar/trumpet etc).

The MobileNet doesn’t output final class labels yet.
It produces a 1024-dimensional vector called an embedding — basically a “numeric fingerprint” of the sound.

Example: 	Elephant sound → [0.23, 0.67, -0.11, ...]
		Monkey sound → [0.91, 0.12, -0.53, ...]

		These embeddings capture what the sound “looks like”.

4)Classification Layer:
these embeddings are passed to fully connected neural layer to predict which sound class it belongs to.
Fully connected layer outputs probabilities across 521 sound classes.

5)Output Filtering:
Extract top-k predictions and map to your target animals.


============================================================
waveform = record_audio(16000)
spectrogram = compute_mel_spectrogram(waveform)
embeddings = MobileNetV1(spectrogram)
scores = DenseLayer(embeddings)
predicted_class = argmax(scores)
display(predicted_class)
============================================================



# --- YAMNet Animal Sound Detection ---
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import numpy as np
import sounddevice as sd
import time

# Paths
MODEL_PATH = "/home/user/wildlife/yamnet/archive"  # downloaded model folder
CSV_PATH = "/home/user/wildlife/yamnet/yamnet_class_map.csv"

# Load model
yamnet_model = hub.load(MODEL_PATH)

# Load class names
class_names = [line.strip().split(',')[0] for line in open(CSV_PATH).readlines()[1:]]

print("🎧 Listening for animal sounds... (Press Ctrl+C to stop)")

SAMPLE_RATE = 16000
DURATION = 3  # seconds

while True:
    print("\nRecording...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

    # Convert to 1D tensor
    waveform = np.squeeze(audio)

    # Run inference
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()

    # Top predictions
    top5 = np.argsort(mean_scores)[::-1][:5]
    print("\nTop Predictions:")
    for i in top5:
        print(f"{class_names[i]}: {mean_scores[i]*100:.2f}%")

    time.sleep(1)

