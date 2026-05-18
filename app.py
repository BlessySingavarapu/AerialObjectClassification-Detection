import os
import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. Load Models ---
@st.cache_resource # Keeps models in memory for speed
def load_models():
    base_dir = 'D:/Python_Code/Projects_In_Course_Time/Project-5_AerialObjectClassification&Detection/'
    best_bird_drone_model_dir = os.path.join(base_dir, 'best_model.keras')
    # Load your saved classification model (Step 7)
    classifier = tf.keras.models.load_model(best_bird_drone_model_dir)
    # Load your trained YOLOv8 model (Step 8)
    #bird_drone_detection = os.path.join(base_dir, 'runs/detect/bird_drone_detection/weights/best.pt')
    #detector = YOLO(bird_drone_detection)
    #return classifier, detector
    return classifier

#classifier, detector = load_models()
classifier = load_models()


# --- 2. Simple UI ---
st.title("🦅 Aerial Object Sentinel")
st.write("Upload an aerial image to classify and detect Birds vs. Drones.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    st.write("---")
    col1, col2 = st.columns(2)

    # --- 3. Run Classification (Step 4 & 6) ---
    with col1:
        st.subheader("Classification Results")
        # Preprocess for CNN (Resize 224x224 & Normalize [0,1])
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        #prediction = classifier.predict(img_array)[0][0]
        prediction = classifier.predict(img_array, verbose=0)[0][0]
        label = "Drone" if prediction > 0.5 else "Bird"
        confidence = prediction if label == "Drone" else (1 - prediction)

        st.metric(label=f"Detected: {label}", value=f"{confidence*100:.2f}% Confidence")

    # --- 4. Run YOLOv8 Detection (Optional Step) ---
    #with col2:
        #st.subheader("Object Detection")
        # Run YOLO inference
        #results = detector(image)

        # Plot results back onto the image
        #res_plotted = results[0].plot()
        #st.image(res_plotted, caption='YOLOv8 Bounding Boxes', use_container_width=True)
