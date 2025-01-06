import streamlit as st
import os
import tempfile
from PIL import Image
from mosaic import mosaic  # Ensure this imports the correct mosaic function

class MosaicApp:
    def __init__(self):
        st.title("Mosaic Image Generator")
        self.main_image_path = None
        self.tiles_directory = None
        self.mosaic_image_path = None  # To store the path of the generated mosaic image

        # Load and display the logo
        self.load_logo()

        # UI Elements
        self.create_widgets()

    def load_logo(self):
        logo_path = "Logo.png"  # Path to your logo image
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((200, 100), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
            st.image(logo_image, caption="Logo", use_column_width=True)

    def create_widgets(self):
        # Load Image Button
        self.main_image_path = st.file_uploader("Load Main Image", type=["jpg", "jpeg", "png"])
        
        # Upload Tiles Images
        tiles_files = st.file_uploader("Upload Tiles Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Opacity Slider
        opacity = st.slider("Opacity Level:", 0, 100, 100)  # Default to 100%

        # Generate Mosaic Button
        if st.button("Generate Mosaic"):
            if self.main_image_path and tiles_files:
                try:
                    # Create a temporary directory to save tiles
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        for tile in tiles_files:
                            tile_path = os.path.join(tmpdirname, tile.name)
                            with open(tile_path, "wb") as f:
                                f.write(tile.getbuffer())
                        
                        # Generate the mosaic and save it to a file
                        mosaic(self.main_image_path, tmpdirname, opacity / 100)  # Pass opacity as a fraction
                        st.success("Mosaic generated successfully!")

                        # Display the newly created mosaic image
                        out_file_path = 'mosaic.jpeg'  # Path to the generated mosaic image
                        if os.path.exists(out_file_path):
                            st.image(out_file_path, caption="Latest Mosaic", use_column_width=True, clamp=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    MosaicApp()