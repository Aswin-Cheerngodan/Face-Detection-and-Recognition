# Face Detection and Recognition 🧠📸

**Real-time face detection and recognition with YOLO, deep learning and Streamlit.**

This helps you **detect faces in images or webcam**, **recognize individuals from a known dataset** — all through a sleek Streamlit interface. Just upload a photo or activate your webcam and see the power of real-time facial recognition.


---

## 🔍 Features

- **Face Detection**: Detect all visible human faces from uploaded images, vidoes or webcam streams using trained YOLOv8.

- **Face Recognition**: Match detected faces against a known set using deep learning.

- **Webcam Support**: Activate webcam directly in the Streamlit app for real-time recognition.

- **Logging & Labeling**: Add and label new faces and log recognition events.

---

## 🧠 Tech Stack

**Backend**: Python, OpenCV, Streamlit  
**Machine Learning**: TensorFlow, ultralytics, numpy, PIL  
**Frontend**: Streamlit  
**Infrastructure**: Docker  

---

## ✅ Prerequisites

- **Docker** (recommended)
- **Python 3.12** (if running locally)
- **Camera** (optional, for webcam input)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Aswin-Cheerngodan/Face-Detection-and-Recognition.git
```

### 2. Run with Docker
```bash
docker build -t face-app .
docker run -d -p 8501:8501 face-app
```

### 3. Run Locally (Without Docker)
```bash
python -m venv myenv
myenv\Scripts\activate  

pip install -r requirements.txt
streamlit run src/main.py
```

### 🌐 Access the App
FaceDetectPro: http://localhost:8501

### 🧪 Example Usage
- Upload an image or video containing faces.  
- App detects and labels known individuals.  
- Activate webcam to start real-time recognition.

### 📬 Contact
Questions or contributions? Open an issue or reach out at aachu8966@gmail.com
