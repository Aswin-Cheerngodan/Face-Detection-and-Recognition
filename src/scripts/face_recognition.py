import numpy as np
import cv2
from tensorflow.keras.models import load_model


model = load_model("artifacts/models/best_vgg_model.keras")
def recognize_face(face_img, class_names=None):
    if class_names is None:
        class_names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachan', 'Andy Samberg',
                       'Anushka Sharma', 'Hrithik Roshan', 'Priyanka Chopra', 'Unknown', 'Vijay Devarakonda', 'Virat Kohli']
    try:
        face_resized = cv2.resize(face_img, (224, 224)) / 255.0
        input_tensor = np.expand_dims(face_resized, axis=0)
        preds = model.predict(input_tensor, verbose=0)[0]
        best_idx = np.argmax(preds)
        best_score = preds[best_idx]
        return class_names[best_idx] if best_score >= 0.9 else "Unknown"
    except Exception as e:
        print(f"Face recognition error: {e}")
        return "Unknown"