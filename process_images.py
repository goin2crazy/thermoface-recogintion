import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from config import load_config

def load_image(image_path):
    """Loads an image (BMP supported) and ensures it's in RGB mode."""
    return Image.open(image_path).convert("RGB")

def crop_image(image, crop_box):
    """Crops the image to remove unnecessary elements."""
    return image.crop(crop_box)

def transform_image(image, img_size=(128, 128)):
    """Resizes, converts to tensor, and normalizes the image."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

def show_image(image):
    """Displays the image using Matplotlib."""
    plt.imshow(image)
    plt.axis("off")  # Hide axis for a cleaner look
    plt.show()

def image_to_tensor(image_path, cfg): 
    
    # Load BMP image directly
    image = load_image(image_path)
    crop_box = (cfg['crop_right'], 
                cfg['crop_up'], 
                image.size[0] - cfg['crop_left'], 
                image.size[1] - cfg['crop_down'])  # Adjust these values as needed
    # Crop the image
    cropped_image = crop_image(image, crop_box)

    # Convert to tensor
    tensor_image = transform_image(cropped_image, img_size=cfg['to_image_size'])
    return {
        "image": cropped_image, 
        "tensor": tensor_image
            }

if __name__ == "__main__": 
    image_path = r"Face recognetion\1\RAIN\img20240303_014006.bmp"  # Adjust if needed
    cfg = load_config()

    data = image_to_tensor(image_path, cfg)

    # Show cropped image
    show_image(data["image"])

    print(f"✅ Processed image shape: {data['tensor'].shape}")  # Should be [3, 128, 128] (RGB)
