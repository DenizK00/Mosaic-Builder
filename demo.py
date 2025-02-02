import streamlit as st
import os
import tempfile
from PIL import Image
from mosaic import mosaic, TileProcessor, TargetImage, compose

class MosaicApp:
    def __init__(self):
        st.set_page_config(page_title="Mosaic Image Generator", layout="wide")
        st.title("Mosaic Image Generator")
        
        # Initialize session state
        if 'mosaic_image' not in st.session_state:
            st.session_state.mosaic_image = None
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = tempfile.mkdtemp()
        
        self.create_sidebar()
        self.create_main_interface()

    def create_sidebar(self):
        with st.sidebar:
            st.header("Settings")
            self.opacity = st.slider("Tile Opacity", 0.0, 1.0, 0.3, 0.1)
            self.size = st.selectbox(
                "Output Size",
                options=["A1", "A2", "A3", "A4"],
                index=0,  # Default to A4
                help="Select the output size for your mosaic"
            )
            st.info("""
            How it works:
            1. Upload a main image
            2. Upload multiple tile images (at least 10)
            3. Adjust opacity if needed
            4. Select output size
            5. Click Generate Mosaic
            """)

    def save_uploaded_file(self, uploaded_file, directory):
        """Save uploaded file to a temporary directory and return the path"""
        if uploaded_file is not None:
            file_path = os.path.join(directory, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if os.path.getsize(file_path) == 0:  # Check if the file is empty
                st.error(f"Uploaded file '{uploaded_file.name}' is empty.")
                return None
            return file_path
        return None

    def create_main_interface(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Images")
            self.main_image = st.file_uploader(
                "Upload Main Image",
                type=["jpg", "jpeg", "png"],
                help="This is the image that will be recreated using tiles"
            )
            
            if self.main_image:
                st.image(self.main_image, caption="Main Image", use_container_width=True)
        
        with col2:
            self.tiles = st.file_uploader(
                "Upload Tile Images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="These images will be used as tiles to create the mosaic (minimum 10 images)"
            )
            
            if self.tiles:
                st.write(f"Uploaded {len(self.tiles)} tile images")
                cols = st.columns(4)
                for idx, tile in enumerate(self.tiles[:4]):
                    with cols[idx]:
                        st.image(tile, caption=f"Tile {idx+1}", use_container_width=True)

        if st.button("Generate Mosaic", type="primary", disabled=not (self.main_image and self.tiles and len(self.tiles) >= 10)):
            if self.main_image and self.tiles:
                with st.spinner("Processing images... This may take a few minutes."):
                    try:
                        # Create temporary directories for main image and tiles
                        tiles_dir = os.path.join(st.session_state.temp_dir, "tiles")
                        os.makedirs(tiles_dir, exist_ok=True)
                        
                        # Save main image
                        main_image_path = self.save_uploaded_file(self.main_image, st.session_state.temp_dir)
                        
                        # Save all tile images
                        for tile in self.tiles:
                            self.save_uploaded_file(tile, tiles_dir)
                        
                        # Generate mosaic using the mosaic.py functions with selected size
                        result_image = mosaic(main_image_path, tiles_dir, self.opacity, size=self.size)
                        
                        if result_image:
                            st.session_state.mosaic_image = result_image
                            st.success("Mosaic generated successfully!")
                        else:
                            st.error("Failed to generate mosaic")
                            
                    except Exception as e:
                        st.error(f"Error during mosaic generation: {str(e)}")

        if st.session_state.mosaic_image:
            st.subheader("Generated Mosaic")
            st.image(st.session_state.mosaic_image, caption="Generated Mosaic", use_container_width=True)
            
            # Prepare image for download
            if isinstance(st.session_state.mosaic_image, Image.Image):
                buf = self.get_image_bytes(st.session_state.mosaic_image)
                
                if st.download_button(
                    label="Download Mosaic",
                    data=buf,
                    file_name="mosaic.jpeg",
                    mime="image/jpeg"
                ):
                    st.success("Download started!")

    @staticmethod
    def get_image_bytes(image):
        """Convert PIL Image to bytes for downloading"""
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

if __name__ == "__main__":
    MosaicApp()