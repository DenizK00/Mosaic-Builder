import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # Import Image and ImageTk from PIL
from mosaic import mosaic  # Assuming mosaic.py is in the same directory

class MosaicApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Mosaic Image Generator")
        self.master.geometry("800x600")

        # Load and display the logo
        self.load_logo()

        # UI Elements
        self.create_widgets()

    def load_logo(self):
        logo_path = "Logo.png"  # Path to your logo image
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((200, 100), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
            self.logo = ImageTk.PhotoImage(logo_image)
            self.logo_label = tk.Label(self.master, image=self.logo)
            self.logo_label.pack(pady=10)  # Add some padding

    def create_widgets(self):
        # Load Image Button
        self.load_image_btn = tk.Button(self.master, text="Load Main Image", command=self.load_image)
        self.load_image_btn.pack(pady=10)

        # Load Tiles Button
        self.load_tiles_btn = tk.Button(self.master, text="Load Tiles Directory", command=self.load_tiles)
        self.load_tiles_btn.pack(pady=10)

        # Opacity Slider
        self.opacity_label = tk.Label(self.master, text="Opacity Level:")
        self.opacity_label.pack(pady=5)
        self.opacity_slider = tk.Scale(self.master, from_=0, to=100, orient=tk.HORIZONTAL)
        self.opacity_slider.set(100)  # Default to 100%
        self.opacity_slider.pack(pady=5)

        # Generate Mosaic Button
        self.generate_btn = tk.Button(self.master, text="Generate Mosaic", command=self.generate_mosaic)
        self.generate_btn.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(self.master, text="")
        self.status_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.main_image_path = file_path
            self.status_label.config(text="Main image loaded.")

    def load_tiles(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.tiles_directory = folder_path
            self.status_label.config(text="Tiles directory loaded.")

    def generate_mosaic(self):
        if hasattr(self, 'main_image_path') and hasattr(self, 'tiles_directory'):
            opacity = self.opacity_slider.get() / 100
            try:
                mosaic(self.main_image_path, self.tiles_directory, opacity)
                self.status_label.config(text="Mosaic generated successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showerror("Error", "Please load both the main image and tiles directory.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MosaicApp(root)
    root.mainloop()
