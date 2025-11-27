from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi

class ImageHandler():

    def __init__(self, root, canvas):
        self.root = root
        self.canvas = canvas
        
        # Get canvas size and create a PIL image and a draw object
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        self.pil_image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.pil_image)
        
        # This is the final processed image that the neural net will use
        self.image = Image.new('L', (28, 28))

    def add_line(self, old_x, old_y, x, y, width, color):
        """Draws a line on the in-memory PIL image."""
        # This check is crucial to avoid drawing a line from (None, None)
        if old_x is not None and old_y is not None:
            self.draw.line([old_x, old_y, x, y], fill=color, width=int(width), joint="curve")

    def clear(self):
        """Clears the in-memory PIL image."""
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="black")
        self.image = Image.new('L', (28, 28))

    def update(self):
        """
        Processes the in-memory PIL image to create the 28x28 centered image
        for the neural network.
        """
        base_img = self.pil_image

        # Converting image to numpy array to crop it
        base_img_data = np.asarray(base_img)
        try:
            non_empty_columns = np.where(base_img_data.max(axis=0) > 0)[0]
            non_empty_rows = np.where(base_img_data.max(axis=1) > 0)[0]
            cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
            
            base_img_data_cropped = base_img_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
            cropped_img = Image.fromarray(base_img_data_cropped)
        except ValueError: # Handle case where image is empty
            self.image = Image.new('L', (28, 28))
            return

        if cropped_img.size[0] <= 1 and cropped_img.size[1] <= 1:
            self.image = Image.new('L', (28, 28))
            return

        # Resizing image to (20, 20) while keeping aspect ratio
        percent = min(20 / float(cropped_img.size[0]), 20 / float(cropped_img.size[1]))
        wsize = int((float(cropped_img.size[0]) * float(percent)))
        hsize = int((float(cropped_img.size[1]) * float(percent)))
        resized_img = cropped_img.resize((wsize, hsize), Image.Resampling.LANCZOS)

        # Finding center of mass of image
        cy, cx = ndi.center_of_mass(resized_img)

        # Creating a (28, 28) image and pasting the old one at the center of the new one
        final_img = Image.new('L', (28, 28))
        final_img.paste(resized_img, (int(final_img.size[0]/2 - round(cx)), int(final_img.size[1]/2 - round(cy))))

        self.image = final_img
    
    def save(self):
        Path("output").mkdir(parents=True, exist_ok=True)
        self.image.save("output/image.jpg", quality=100)