import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model


# Load your trained model
model = load_model("hwd_recognition_model.h5")

st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read file bytes into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess image
    img_resized = cv2.resize(image, (28,28))       # resize to 28x28
    img_resized = img_resized / 255.0              # normalize
    img_reshaped = img_resized.reshape(1,28,28,1)  # reshape for model

    # Prediction
    y_prob = model.predict(img_reshaped)
    pred_class = np.argmax(y_prob, axis=-1)[0]

    st.subheader(f"✅ Predicted Digit: {pred_class}")
    # st.bar_chart(y_prob[0])   # probability distribution

