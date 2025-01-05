# app.py
from flask import Flask, request, render_template, send_from_directory, url_for
from PIL import Image
import os
from mosaic import mosaic  # Assuming mosaic.py is in the same directory
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')  # Create an index.html file for the UI

@app.route('/generate', methods=['POST'])
def generate_mosaic():
    main_image = request.files['main_image']
    tiles_directory_files = request.files.getlist('tiles_directory')  # Get list of files
    opacity = float(request.form['opacity']) / 100

    # Ensure the uploads directory exists
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)

    if main_image and tiles_directory_files:
        main_image_path = os.path.join(uploads_dir, main_image.filename)
        main_image.save(main_image_path)

        # Save tiles to a temporary directory
        tiles_directory = os.path.join(uploads_dir, 'tiles')  # Create a directory to store tiles
        os.makedirs(tiles_directory, exist_ok=True)

        for tile in tiles_directory_files:
            tile_path = os.path.join(tiles_directory, tile.filename)
            tile.save(tile_path)

        try:
            # Generate the mosaic and get the image object
            mosaic_image = mosaic(main_image_path, tiles_directory, opacity)

            # Check if the mosaic image was created successfully
            if mosaic_image is None:
                return "Error: Mosaic image could not be created."

            # Save the mosaic image to the root directory
            output_image_path = 'mosaic.jpeg'  # Path to save in the root directory
            mosaic_image.save(output_image_path)

            # Return the URL for the generated image in the root directory
            return render_template('result.html', image_url=url_for('static', filename='mosaic.jpeg'))
        except Exception as e:
            return str(e)
    return "Please upload both the main image and tiles directory."

@app.route('/uploads/<path:filename>')
def send_uploads(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Allow external requests