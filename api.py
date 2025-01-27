from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from typing import List
from PIL import Image
from mosaic import mosaic
import time

app = FastAPI()


# Allow CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def cleanup_files(temp_dir: str):
    """Cleanup temporary files after response is sent"""
    try:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Cleanup error: {str(e)}")


@app.post("/generate_mosaic")
async def generate_mosaic(
    background_tasks: BackgroundTasks,
    main_image: UploadFile = File(...),
    tile_images: List[UploadFile] = File(...),
    opacity: float = Form(default=0.3),
    size: str = Form(default="A4")
):
    # Validate main image
    if not main_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Main file must be an image")
    
    # Validate tile images
    for tile in tile_images:
        if not tile.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="All tile files must be images")
    
    # Create a unique temporary directory for each request
    temp_dir = tempfile.mkdtemp(prefix="mosaic_")
    tiles_dir = os.path.join(temp_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    try:
        # Save the main image
        main_image_path = os.path.join(temp_dir, "main_image.jpg")
        content = await main_image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty main image file")
        with open(main_image_path, "wb") as f:
            f.write(content)

        # Save all tile images
        for tile_image in tile_images:
            content = await tile_image.read()
            if not content:
                continue  # Skip empty files
            tile_path = os.path.join(tiles_dir, tile_image.filename)
            with open(tile_path, "wb") as f:
                f.write(content)

        # Check if we have enough tile images
        if len(os.listdir(tiles_dir)) < 1:
            raise HTTPException(status_code=400, detail="No valid tile images provided")

        # Generate a unique output filename using a timestamp
        timestamp = int(time.time())
        output_filename = f"mosaic_{timestamp}.jpeg"
        output_path = os.path.join(temp_dir, output_filename)

        # Generate the mosaic image
        try:
            result_image = mosaic(main_image_path, tiles_dir, opacity, output_path, size)
            print(f"Saving mosaic to {output_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mosaic generation failed: {str(e)}")

        # Check if the file was created
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Generated file does not exist")

        # Add cleanup as a background task
        background_tasks.add_task(cleanup_files, temp_dir)
        
        return FileResponse(
            output_path,
            media_type='image/jpeg',
            filename=output_filename
        )

    except Exception as e:
        # Clean up files immediately in case of error
        await cleanup_files(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)