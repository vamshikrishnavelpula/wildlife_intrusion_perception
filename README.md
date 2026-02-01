# 🐾 Wildlife Intrusion Perception System (YOLO + YAMNet)

## 📌 Overview
The **Wildlife Intrusion Perception System** is an AI-based solution designed to detect and identify wildlife intrusions near human-inhabited or agricultural areas.  
The system combines **computer vision** and **audio-based classification** to improve detection accuracy and reduce false alarms.

- **YOLO** is used for real-time animal detection from video streams.
- **YAMNet** is used for audio-based animal sound classification.

This multi-modal approach helps in early warning and prevention of human–wildlife conflicts.

---

## 🚀 Key Features
- Real-time wildlife detection using video input
- Animal sound classification using audio signals
- Combines **visual + audio intelligence**
- Supports live camera feed and recorded media
- Reduces false positives compared to single-model systems
- Scalable for forest borders, farms, and highways

---

## 🧠 Technologies Used

### 🔹 Programming Language
- Python

### 🔹 Computer Vision
- YOLO (You Only Look Once)
- OpenCV

### 🔹 Audio Classification
- YAMNet (TensorFlow-based audio event classifier)

### 🔹 Libraries & Frameworks
- NumPy
- TensorFlow
- PyTorch (for YOLO)
- Librosa
- Scikit-learn

---

## ⚙️ System Architecture
1. **Video Input**
   - Live camera or video file
   - YOLO detects animals in each frame

2. **Audio Input**
   - Microphone or audio file
   - YAMNet classifies animal sounds

3. **Decision Module**
   - Combines video and audio predictions
   - Confirms wildlife intrusion

4. **Output**
   - Detection logs
   - Alerts (visual/audio)

---

## 🔄 Workflow
1. Capture video frames and audio signals
2. Detect animals using YOLO
3. Classify sounds using YAMNet
4. Fuse results for accurate intrusion detection
5. Display detection results and alerts

---

## 📂 Project Structure
wildlife-intrusion-perception/
├── yamnet/
├── yolov12algo.py
├── yamnet_live.py
├── yamnet_test.py
├── fix_labels.py
├── README.md
├── requirements.txt
└── .gitignore


---

## 🖥️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vamshikrishnavelpula/wildlife_intrusion_perception.git
cd wildlife_intrusion_perception

pip install -r requirements.txt

python yamnet_live.py
