# 🛰️ Aerial Object Classification & Detection System

## 📌 Overview

This project presents a deep learning-based system capable of **classifying and detecting aerial objects** as either **Birds 🐦 or Drones 🚁**.

It combines:

* **Image Classification** using Transfer Learning (MobileNetV2)
* **Object Detection** using YOLOv8
* **Interactive Web Interface** using Streamlit

The system is designed for real-world applications such as **airspace monitoring, wildlife protection, and surveillance systems**.

---

## 🚀 Features

* ✅ Binary Image Classification (Bird vs Drone)
* 📦 Real-time Object Detection with Bounding Boxes
* 🎯 Confidence Score Visualization
* 🌐 Interactive Web App (Streamlit)
* 📊 Model Evaluation (Confusion Matrix, F1 Score)
* ⚡ Lightweight and Fast Inference

---

## 🧠 Tech Stack

| Category         | Tools                |
| ---------------- | -------------------- |
| Deep Learning    | TensorFlow, Keras    |
| Object Detection | YOLOv8 (Ultralytics) |
| Frontend         | Streamlit            |
| Data Processing  | NumPy, OpenCV        |
| Visualization    | Matplotlib, Sklearn  |

---

## 🏗️ Project Architecture

```text
Input Image
     ↓
[Classification Model - MobileNetV2]
     ↓
Prediction (Bird / Drone)
     ↓
[YOLOv8 Detection Model]
     ↓
Bounding Boxes + Labels + Confidence
     ↓
Streamlit UI Output
```

---

## 📂 Project Structure

```text
Aerial Detection Project/
│
├── dataset/                 # Classification dataset
├── yolo_dataset/            # Detection dataset (YOLO format)
├── runs/                    # YOLO training outputs
│
├── train.py                 # Classification training
├── evaluate.py              # Model evaluation
├── yolo_train.py            # YOLO training
├── yolo_predict.py          # Detection testing
├── app.py                   # Streamlit web app
│
├── best_model.h5
├── final_model.h5
├── requirements.txt
└── README.md
```

---

## 📊 Results

### 🔹 Classification Model

* Accuracy: **~98%**
* High Precision & Recall across both classes

### 🔹 YOLOv8 Detection

* Detects objects with bounding boxes
* Confidence scores range: **0.4 – 0.9+**
* Real-time detection capability

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train Classification Model

```bash
python train.py
```

### 3️⃣ Evaluate Model

```bash
python evaluate.py
```

### 4️⃣ Train YOLO Model

```bash
python yolo_train.py
```

### 5️⃣ Run Web App

```bash
python -m streamlit run app.py
```

---

## 📸 Sample Output

* 🐦 Bird detected with bounding box
* 🚁 Drone detected with confidence score
* 📦 Multiple object detection supported

---

## 🎯 Real-World Applications

* ✈️ Airport Bird Strike Prevention
* 🛡️ Defense & Surveillance Systems
* 🌿 Wildlife Monitoring
* 🚁 Drone Detection in Restricted Zones

---

## ⚠️ Limitations

* Detection confidence may vary for small or unclear objects
* Performance depends on dataset quality and training epochs
* CPU-based training is slower compared to GPU

---

## 🔮 Future Improvements

* 🎥 Real-time video detection
* ☁️ Cloud deployment
* 📱 Mobile app integration
* 🧠 Model optimization for higher accuracy

---

## 👨‍💻 Author

**Yash Yevale**
AI/ML Developer | Web Developer

---

## ⭐ Conclusion

This project demonstrates a **complete end-to-end AI pipeline**, combining classification, detection, and deployment into a real-world usable system.

> From raw images → to intelligent detection → to user-friendly application.
