import argparse
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

# Assuming these are defined elsewhere in your project
from model import FaceEmbeddingModel, Generator
from config import load_config  # Your configuration loader

def load_image(image_path, transform=None):
    """Loads an image and applies a transformation if provided."""
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    return image

def generator_inference(image_path, face_model, generator, device, visualization=False):
    """Performs generator inference: computes an embedding and passes it to the generator."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
    image = load_image(image_path, transform)
    image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        embedding = face_model(image_tensor)
        generated = generator(embedding)

    # Convert generated tensor to a numpy image for plotting
    generated_image = generated.squeeze(0).cpu()  # remove batch dimension
    generated_image_np = generated_image.permute(1, 2, 0).numpy()

    if visualization:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(image_path))
        plt.title("Input Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(generated_image_np)
        plt.title("Generated Image")
        plt.axis("off")
        plt.show()
    else:
        print("Generated image tensor shape:", generated.shape)
    return generated_image

def embedding_test(query_image_path, compare_image_paths, face_model, device, visualization=False):
    """Finds the most relevant image by comparing embeddings."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
    query_image = load_image(query_image_path, transform)
    query_tensor = query_image.unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = face_model(query_tensor)

    if visualization: 
        print(query_embedding)
        
    best_distance = float("inf")
    best_image_path = None
    for cmp_path in compare_image_paths:
        cmp_image = load_image(cmp_path, transform)
        cmp_tensor = cmp_image.unsqueeze(0).to(device)
        with torch.no_grad():
            cmp_embedding = face_model(cmp_tensor)
        # Calculate Euclidean distance
        distance = torch.norm(query_embedding - cmp_embedding, p=2).item()
        if distance < best_distance:
            best_distance = distance
            best_image_path = cmp_path

    print(f"Best matching image: {best_image_path} with distance {best_distance:.4f}")
    
    if visualization:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(query_image_path))
        plt.title("Query Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(Image.open(best_image_path))
        plt.title("Best Match")
        plt.axis("off")
        plt.show()
    return best_image_path, best_distance

def main():
    parser = argparse.ArgumentParser(
        description="Inference script for generator inference and embedding test."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--inference-type", type=str, required=True, choices=["generator-inference", "embedding-test"],
        help="Type of inference to perform."
    )
    parser.add_argument(
        "--visualization", action="store_true",
        help="If set, shows matplotlib visualization of results."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the model checkpoint. If not provided, will use default from config."
    )
    parser.add_argument(
        "--compare-images", type=str, nargs="+",
        help="List of image paths to compare with (required for embedding-test)."
    )
    parser.add_argument(
        "--config-path", type=str, default="", 
        help="Path to the config file. If empty or invalid, uses 'config.yaml' in the current directory."
    )

    args = parser.parse_args()

    # Determine config file path
    config_path = args.config_path.strip()
    if not config_path or not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        print(f"Using default config: {config_path}")

    # Load configuration
    cfg = load_config(config_path)
    print(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    face_model = FaceEmbeddingModel(cfg["model_name"], cfg["embedding_size"]).to(device)
    generator = Generator(cfg["embedding_size"]).to(device)

    # Load checkpoint if available
    checkpoint_path = args.checkpoint if args.checkpoint is not None else cfg.get("inference_checkpoint", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        face_model.load_state_dict(checkpoint["face_model_state"])
        generator.load_state_dict(checkpoint["generator_state"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No valid checkpoint provided, using model initialization.")

    if args.inference_type == "generator-inference":
        generator_inference(args.image, face_model, generator, device, args.visualization)
    elif args.inference_type == "embedding-test":
        if not args.compare_images:
            parser.error("--compare-images is required for embedding-test inference.")
        embedding_test(args.image, args.compare_images, face_model, device, args.visualization)

if __name__ == "__main__":
    main()
