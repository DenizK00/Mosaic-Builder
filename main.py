import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
import numpy as np

class MosaicBuilder:
    def __init__(self, master):
        self.master = master
        self.master.title("Mosaic Image Builder")
        self.image_path = None

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.create_button = tk.Button(master, text="Create Mosaic", command=self.create_mosaic)
        self.create_button.pack()

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        # Load the main image
        self.main_image = Image.open(self.image_path).convert('RGB')  # Convert to RGB
        self.main_image_arr = np.asarray(self.main_image)
        return self.main_image_arr

    def create_mosaic(self):
        # Load small images from a specified directory
        small_images = self.load_small_images()  # New method to load small images
        if not small_images:
            print("No small images found.")
            return

        # Resize the main image to fit the mosaic
        main_image_resized = self.main_image.resize((500, 500))  # Resize to a fixed size for simplicity
        mosaic_width, mosaic_height = main_image_resized.size

        # Create a new blank image for the mosaic
        mosaic_image = Image.new('RGB', (mosaic_width, mosaic_height))

        # Logic to fit small images into the mosaic
        small_image_size = (50, 50)  # Size of each small image
        for y in range(0, mosaic_height, small_image_size[1]):
            for x in range(0, mosaic_width, small_image_size[0]):
                # Select a random small image
                small_image = small_images[(x // small_image_size[0] + y // small_image_size[1]) % len(small_images)]
                small_image_resized = small_image.resize(small_image_size)
                mosaic_image.paste(small_image_resized, (x, y))

        # Save or display the mosaic image
        mosaic_image.show()  # Display the mosaic image

    def load_small_images(self):
        # Logic to load small images from a directory
        small_images = []
        # Example: Load images from a folder (you can customize this)
        folder_path = filedialog.askdirectory()  # Ask user for directory
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                small_images.append(Image.open(img_path))
        return small_images

    
if __name__ == "__main__":
    root = tk.Tk()
    app = MosaicBuilder(root)
    root.mainloop()
