from PIL import Image
import os

def resize_images(input_folder, output_folder, width, height):
    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Open the image file
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Resize the image
                resized_img = img.resize((width, height), Image.ANTIALIAS)
                
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f"Resized {filename} to {width}x{height} and saved as {output_path}")

if __name__ == "__main__":
    # Input and output folders
    input_folder = "/path/to/input/folder"
    output_folder = "/path/to/output/folder"

    # Target width and height
    target_width = 800
    target_height = 600

    # Resize images
    resize_images(input_folder, output_folder, target_width, target_height)
