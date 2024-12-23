import tkinter as tk
from tkinter import filedialog
from PIL import Image

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
        self.main_image = Image.open(self.image_path)

    def create_mosaic(self):
        # Logic to create a mosaic from the main image
        # This is where you would fit the small images
        # For now, we will just print a message
        print("Creating mosaic...")

if __name__ == "__main__":
    root = tk.Tk()
    app = MosaicBuilder(root)
    root.mainloop()
