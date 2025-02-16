import os
from torch.utils.data import Dataset
from typing import List, Dict
from PIL import Image

from config import load_config  # Load metadata
from process_images import image_to_tensor

def find_images(base_path: str, spec: List[str]) -> List[Dict[str, str]]:
    """
    Finds all images inside specified subfolders (case-insensitive, partial matches allowed) 
    and stores them with their unique ID.

    Args:
        base_path (str): The main folder containing subdirectories.
        spec (List[str]): List of subfolder names to search for images.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with {"unique_id": id, "image_path": path}.
    """
    dataset = []
    
    if not spec:  # Skip if spec is empty
        return dataset

    spec = [s.lower() for s in spec]  # Normalize spec names to lowercase

    for unique_id in os.listdir(base_path):  # Iterate over folders (1, 2, etc.)
        unique_path = os.path.join(base_path, unique_id)
        if not os.path.isdir(unique_path):
            continue  # Skip if not a directory

        for folder in os.listdir(unique_path):  # Check actual subfolder names
            if any(s in folder.lower() for s in spec):  # Match ignoring case & allowing substrings
                folder_path = os.path.join(unique_path, folder)

                images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) 
                          if img.lower().endswith(('.png', '.jpg', '.bmp'))]

                for img_path in images:
                    dataset.append({"unique_id": unique_id, "image_path": img_path})  # Store every image separately

    return dataset

class CustomImageDataset(Dataset):
    """
    PyTorch Dataset to load images from "RAIN" folders and convert them to tensors.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_path = cfg["images_dir"]  # Get path from config
        self.data = find_images(self.base_path, cfg['thermo_configuration'])  # Load all images

    def __len__(self):
        """Returns total number of images."""
        return len(self.data)

    def __getitem__(self, index):
        """Loads and returns an image sample and its tensor."""
        sample = self.data[index]  # Get one image entry
        unique_id = sample["unique_id"]
        image_path = sample["image_path"]
        
        processed_data = image_to_tensor(image_path, self.cfg)  # Convert image
        
        return {
            "unique_id": unique_id,  
            "tensor": processed_data["tensor"]  # PyTorch tensor
        }


if __name__ == "__main__": 
    # Load config
    cfg = load_config()

    # Create dataset
    dataset = CustomImageDataset(cfg)

    # Example usage:
    sample = dataset[140]
    print(f"Sample ID: {sample['unique_id']}, Tensor Shape: {sample['tensor'].shape}")