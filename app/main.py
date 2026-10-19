"""Streamlit interface for uploading an image and predicting cat or dog."""

import streamlit as st

from app.image_utils import InvalidImageError, decode_image
from app.predictor import CatDogPredictor, load_predictor


@st.cache_resource
def get_predictor(version: str) -> CatDogPredictor:
    """Cache each loaded model so Streamlit reruns do not reload its weights."""
    return load_predictor(version)


st.set_page_config(page_title="Cat or Dog?", page_icon="🐾")
st.title("Cat or Dog?")
st.write(
    "Upload a JPEG or PNG image and let the model predict whether it shows a cat or dog."
)

model_version = st.selectbox("Model version", ["v2", "v1"], index=0)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = decode_image(uploaded_file.getvalue())
    except InvalidImageError as exc:
        st.error(str(exc))
    else:
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Classify", type="primary"):
            try:
                predictor = get_predictor(model_version)
                probabilities = predictor.predict(image)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(f"Could not run the selected model: {exc}")
            else:
                predicted_class = max(probabilities, key=probabilities.get)
                confidence = probabilities[predicted_class]
                st.success(f"Prediction: {predicted_class.title()}")
                st.metric("Confidence", f"{confidence:.1%}")
                st.write("Class probabilities")
                for class_name, probability in probabilities.items():
                    st.progress(
                        probability, text=f"{class_name.title()}: {probability:.1%}"
                    )
