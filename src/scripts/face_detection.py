from ultralytics import YOLO
from PIL import Image
import numpy as np

yolo = YOLO("artifacts/models/best.pt")

def detect_face(uploaded_file):
    image = Image.open(uploaded_file)
    rgb_image = image.convert("RGB")
    img = np.array(rgb_image)
    results = yolo(img)
    faces = results[0].boxes.xyxy.cpu().numpy().astype(int)
    return faces, img

def detect_face_video(frame):
    results = yolo(frame)
    faces = results[0].boxes.xyxy.cpu().numpy().astype(int)
    return faces, frame


