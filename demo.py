import streamlit as st
import os
import tempfile
from PIL import Image
from mosaic import mosaic, TileProcessor, TargetImage

# Cache the tile processing since it's computationally expensive
@st.cache_data
def process_tiles(tiles_files):
    """Process and cache tile images"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save uploaded files to temporary directory
        for tile in tiles_files:
            tile_path = os.path.join(tmpdirname, tile.name)
            with open(tile_path, "wb") as f:
                f.write(tile.getbuffer())
        
        # Process tiles using TileProcessor
        processor = TileProcessor(tmpdirname)
        return processor.get_tiles()

# Cache the target image processing
@st.cache_data
def process_target_image(image_file):
    """Process and cache the target image"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(image_file.getbuffer())
        target = TargetImage(tmp_file.name)
        result = target.get_data()
        os.unlink(tmp_file.name)  # Clean up the temporary file
        return result

# Cache the mosaic generation
@st.cache_data
def generate_mosaic(target_image_data, tiles_data, opacity=0.3):
    """Generate and cache the mosaic image"""
    try:
        from mosaic import compose, MosaicImage
        
        # Create mosaic image object
        mosaic_img = MosaicImage(target_image_data[0])
        
        # Generate the mosaic and return the image
        return compose(target_image_data, tiles_data, opacity)  # Return the generated mosaic image
    except Exception as e:
        st.error(f"Error generating mosaic: {str(e)}")
        return None

class MosaicApp:
    def __init__(self):
        st.set_page_config(page_title="Mosaic Image Generator", layout="wide")
        st.title("Mosaic Image Generator")
        
        # Initialize session state if needed
        if 'mosaic_image' not in st.session_state:
            st.session_state.mosaic_image = None
        
        self.create_sidebar()
        self.create_main_interface()

    def create_sidebar(self):
        with st.sidebar:
            st.header("Settings")
            self.opacity = st.slider("Tile Opacity", 0.0, 1.0, 0.3, 0.1)
            
            # Add info about the process
            st.info("""
            How it works:
            1. Upload a main image
            2. Upload multiple tile images
            3. Adjust opacity if needed
            4. Click Generate Mosaic
            """)

    def create_main_interface(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Images")
            # Main image upload
            self.main_image = st.file_uploader(
                "Upload Main Image",
                type=["jpg", "jpeg", "png"],
                help="This is the image that will be recreated using tiles"
            )
            
            # Display uploaded image
            if self.main_image:
                st.image(self.main_image, caption="Main Image", use_container_width=True)
        
        with col2:
            # Tiles upload
            self.tiles = st.file_uploader(
                "Upload Tile Images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="These images will be used as tiles to create the mosaic"
            )
            
            if self.tiles:
                st.write(f"Uploaded {len(self.tiles)} tile images")
                # Display a sample of tiles
                cols = st.columns(4)
                for idx, tile in enumerate(self.tiles[:4]):  # Show first 4 tiles
                    with cols[idx]:
                        st.image(tile, caption=f"Tile {idx+1}", use_container_width=True)

        # Generate button
        if st.button("Generate Mosaic", type="primary", disabled=not (self.main_image and self.tiles)):
            if self.main_image and self.tiles:
                with st.spinner("Processing images..."):
                    # Process target image
                    target_data = process_target_image(self.main_image)
                    
                    # Process tiles
                    tiles_data = process_tiles(self.tiles)
                    
                    if tiles_data and tiles_data[0]:
                        # Generate mosaic
                        st.session_state.mosaic_image = generate_mosaic(
                            target_data,
                            tiles_data,
                            self.opacity
                        )
                        
                        if st.session_state.mosaic_image:
                            st.success("Mosaic generated successfully!")
                        else:
                            st.error("Failed to generate mosaic")
                    else:
                        st.error("No valid tiles were processed")

        # Display the mosaic if it exists
        if st.session_state.mosaic_image:
            st.subheader("Generated Mosaic")
            st.image(st.session_state.mosaic_image, caption="Generated Mosaic", use_container_width=True)
            
            # Add download button
            if st.download_button(
                label="Download Mosaic",
                data=self.get_image_bytes(st.session_state.mosaic_image),
                file_name="mosaic.png",
                mime="image/png"
            ):
                st.success("Download started!")

    @staticmethod
    def get_image_bytes(image):
        """Convert PIL Image to bytes for downloading"""
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

if __name__ == "__main__":
    MosaicApp()