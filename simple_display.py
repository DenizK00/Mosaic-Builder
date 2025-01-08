import streamlit as st
from PIL import Image
import os

def main():
    st.title("Mosaic Image Display")

    # Check if the mosaic image exists
    if os.path.exists('mosaic.jpeg'):
        # Open and display the image
        mosaic_image = Image.open('mosaic.jpeg')
        st.image(mosaic_image, caption="Generated Mosaic", use_container_width=True)
    else:
        st.error("Mosaic image not found. Please ensure 'mosaic.jpeg' exists in the root directory.")

if __name__ == "__main__":
    main()