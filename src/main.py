import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import tempfile
import time
from datetime import datetime
from scripts.face_detection import detect_face, detect_face_video
from scripts.face_recognition import recognize_face
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os

# Simulated training history for plotting (replace with your actual model training history)
training_history = {
    'accuracy': [0, 0.056, 0.39, 0.40, 0.66, 0.63, 0.66, 0.81, 0.79, 0.76, 0.76, 0.75, 0.75, 0.80, 0.83, 0.86, 0.84, 0.85, 0.83, 0.88, 0.84, 0.84, 0.84]
}

# Function to plot model accuracy
def plot_accuracy():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(training_history['accuracy'], label='Accuracy', color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Over Epochs')
    ax.legend()
    ax.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Function to process and display frame
def process_frame(frame):
    faces, processed_frame = detect_face_video(frame)
    for x1, y1, x2, y2 in faces:
        face_img = processed_frame[y1:y2, x1:x2]
        name = recognize_face(face_img)
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(processed_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return processed_frame, len(faces)

# Streamlit app configuration
st.set_page_config(page_title="Face Detection & Recognition", layout="centered")
st.title("Face Detection & Recognition")
st.write("Upload an image or select a video source for real-time face detection and recognition.")

# Sidebar for input selection
st.sidebar.header("Input Source")
input_type = st.sidebar.radio("Select input type:", ("Upload Image", "Video Stream"))

# Initialize session state
if "video_running" not in st.session_state:
    st.session_state.video_running = False
if "cap" not in st.session_state:
    st.session_state.cap = None

def release_video():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.video_running = False
    if "tfile" in st.session_state and st.session_state.tfile:
        try:
            os.unlink(st.session_state.tfile.name)
            st.session_state.tfile = None
        except Exception as e:
            print(f"Failed to delete temp file: {e}")


# Main content based on input type
if input_type == "Upload Image":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and process image
        
        faces, processed_img = detect_face(uploaded_file)
        # Get image height (or width) to scale elements proportionally
        h, w = processed_img.shape[:2]
        scale_factor = max(h, w) / 1000  # You can tweak the denominator for better control

        # Draw rectangle with dynamic thickness
        thickness = int(2 * scale_factor)
        font_scale = 0.8 * scale_factor
        
        for x1, y1, x2, y2 in faces:
            face_img = processed_img[y1:y2, x1:x2]
            name = recognize_face(face_img)
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), (255, 0, 0), thickness)
            cv2.putText(processed_img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        st.image(processed_img, channels="RGB", caption=f"Processed Image ({len(faces)} faces detected)")
        st.success(f"Detected {len(faces)} face(s)")

else:
    st.header("Video Stream")
    camera_option = st.selectbox("Choose video source:", ["Webcam", "Video File"])
    
    video_placeholder = st.empty()
    
    if camera_option == "Video File":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile.close()
            st.session_state.tfile = tfile  # Save tfile in session state
            st.session_state.cap = cv2.VideoCapture(tfile.name)
            st.video(video_file)  # Display original video for reference
    else:
        st.write("Using Webcam as video source")
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Video Stream")
    with col2:
        stop_button = st.button("Stop Video Stream")

    if start_button and not st.session_state.video_running:
        if camera_option == "Webcam" and st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)  # Open webcam
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.error("Failed to open video source. Please check your webcam or video file.")
        else:
            st.session_state.video_running = True

    if stop_button:
        release_video()

    # Process video stream
    if st.session_state.video_running and st.session_state.cap is not None:
        try:
            while st.session_state.video_running and st.session_state.cap.isOpened():
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.warning("End of video or failed to read frame.")
                    release_video()
                    break
                
                # Process frame
                processed_frame, num_faces = process_frame(frame)
                
                # Display frame
                video_placeholder.image(
                    processed_frame,
                    channels="BGR",
                    caption=f"Detected {num_faces} face(s)"
                )
                
                # Control frame rate
                time.sleep(0.03)
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            release_video()
        finally:
            if camera_option == "Video File" and 'tfile' in locals():
                os.unlink(tfile.name)  # Clean up temporary file

# Display accuracy plot
st.subheader("Model Training Accuracy")
accuracy_plot = plot_accuracy()
st.image(accuracy_plot, caption="Accuracy vs. Epochs", use_container_width=True)

# Clean up on app exit
def cleanup():
    release_video()

if __name__ == "__main__":
    try:
        st.session_state.cleanup = cleanup
    except:
        release_video()