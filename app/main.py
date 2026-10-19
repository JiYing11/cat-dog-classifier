"""Streamlit interface for the cat and dog image classifier."""

import streamlit as st

from app.image_utils import InvalidImageError, decode_image


st.set_page_config(page_title="Cat or Dog?", page_icon="🐾")
st.title("Cat or Dog?")
st.write("Upload a JPEG or PNG image to get it ready for classification.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = decode_image(uploaded_file.getvalue())
    except InvalidImageError as exc:
        st.error(str(exc))
    else:
        st.image(image, caption="Image ready for classification", use_container_width=True)
        st.info("The image upload is working. Model training and prediction will be added next.")
