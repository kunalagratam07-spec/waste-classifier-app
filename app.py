import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Smart Waste Classifier", layout="centered")

# Global spacing + style
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00ffcc;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("baseline_best.keras")

# Classes (must match training order)
class_names = ['glass', 'metal', 'organic', 'paper', 'plastic']

# Title
st.markdown('<div class="title">♻️ Smart Waste Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let AI classify the waste type</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📷 Uploaded Image", width=300)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    with col2:
        st.markdown("### 🧠 Prediction")

        if predicted_class == "organic":
            st.warning(f"🌱 {predicted_class.upper()}")
        elif predicted_class == "plastic":
            st.success(f"♻️ {predicted_class.upper()}")
        else:
            st.info(f"{predicted_class.upper()}")

        st.markdown("### 🔥 Confidence")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}%")

    # Divider
    st.markdown("---")

    # Probability chart
    st.markdown("### 📊 Prediction Breakdown")

    prob_dict = {
        class_names[i]: float(prediction[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(prob_dict)

# Footer
st.markdown("---")