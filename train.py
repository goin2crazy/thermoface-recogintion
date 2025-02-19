import os 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import CustomImageDataset  # Import your dataset class
from dataloader import * 
from config import load_config
from model import FaceEmbeddingModel, Generator  # Import models
from utils import * 

import torch.nn as nn
from tqdm import tqdm

def train_step(face_model, generator, batch, optimizer_face, optimizer_gen, device):
    """
    Performs one training step.
    
    Args:
        face_model (nn.Module): The embedding model (e.g. modified resnet50d).
        generator (nn.Module): The generator model that reconstructs images from embeddings.
        batch (dict): Dictionary with at least key 'tensor' containing images 
                      (shape: [batch_size, 3, 224, 224]).
        optimizer_face (torch.optim.Optimizer): Optimizer for the face model.
        optimizer_gen (torch.optim.Optimizer): Optimizer for the generator.
        device (torch.device): Device to run the computations on.
    
    Returns:
        tuple: (embedding_loss, generator_loss)
            embedding_loss: Loss for the face model (variance loss + reconstruction MSE).
            generator_loss: Loss for the generator (MSE between generated and original images).
    """
    # Set models to training mode
    face_model.train()
    generator.train()
    
    # Move images to device
    images = batch['tensor'].to(device)  # shape: [B, 3, 224, 224]
    
    # ====================
    # Update Face Model (Embedding Network)
    # ====================
    optimizer_face.zero_grad()
    
    # Forward pass through face model to get embeddings
    embeddings = face_model(images)  # shape: [B, embedding_size]
    
    # Compute variance loss: encourages embeddings in the same batch (same ID) to be similar.
    # We use the mean squared distance from the batch mean.
    mean_embedding = embeddings.mean(dim=0, keepdim=True)
    variance_loss = ((embeddings - mean_embedding) ** 2).mean()
    
    # Compute reconstruction loss for the face model:
    # We want the embeddings to be good for reconstructing the original image.
    # Use the generator, but detach its parameters so that only the face model is updated here.
    generated_images_for_face = generator(embeddings).detach()
    mse_loss_face = nn.MSELoss()(generated_images_for_face, images)
    
    # Total embedding loss: you might consider weighting the variance loss if needed.
    embedding_loss = variance_loss + mse_loss_face  # You could do: lambda1*variance_loss + lambda2*mse_loss_face
    embedding_loss.backward()
    optimizer_face.step()
    
    # ====================
    # Update Generator Model
    # ====================
    optimizer_gen.zero_grad()
    
    # For generator update, we want to update generator parameters only,
    # so we detach the embeddings (to not update the face model).
    embeddings_for_gen = face_model(images).detach()
    generated_images = generator(embeddings_for_gen)
    mse_loss_gen = nn.MSELoss()(generated_images, images)
    mse_loss_gen.backward()
    optimizer_gen.step()
    
    return embedding_loss.item(), mse_loss_gen.item()

def test_train_step(): 
        # Load config
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    face_model = FaceEmbeddingModel(cfg["model_name"], cfg["embedding_size"]).to(device)
    generator = Generator(cfg["embedding_size"]).to(device)

    # Optimizers
    optimizer_face = optim.Adam(face_model.parameters(), lr=1e-4)
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-4)

    # Create dataset & dataloader
    dataset = CustomImageDataset(cfg)  # Load dataset (assumes images in RAIN folder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Batch size = 4 for testing

    # Get a batch and test train_step
    batch = next(iter(dataloader))  # Get one batch
    embedding_loss, gen_loss = train_step(face_model, generator, batch, optimizer_face, optimizer_gen, device)

    # Print losses to check everything works
    print(f"Embedding Loss: {embedding_loss:.4f}")
    print(f"Generator Loss: {gen_loss:.4f}")


def save_checkpoint(cfg, epoch, face_model, generator, optimizer_face, optimizer_gen, loss_history, val_loss, early_stopper):
    """
    Saves checkpoints based on the 'save_method' parameter in the config.
    
    Args:
        cfg (dict): Configuration dictionary.
        epoch (int): Current training epoch.
        face_model (torch.nn.Module): Face embedding model.
        generator (torch.nn.Module): Generator model.
        optimizer_face (torch.optim.Optimizer): Optimizer for face model.
        optimizer_gen (torch.optim.Optimizer): Optimizer for generator.
        loss_history (list): List of training loss history.
        val_loss (float): Validation loss (sum of embedding and generator losses).
        early_stopper (EarlyStopping): Early stopping handler.
    """
    save_method = cfg.get("save_method", "all")
    checkpoint_data = {
        "epoch": epoch,
        "face_model_state": face_model.state_dict(),
        "generator_state": generator.state_dict(),
        "optimizer_face_state": optimizer_face.state_dict(),
        "optimizer_gen_state": optimizer_gen.state_dict(),
        "loss_history": loss_history  # Save loss history for analysis
    }

    if save_method == "all":
        checkpoint_path = f"{cfg['to_save_checkpoint']}_{epoch}.pth"
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    elif save_method == "main":
        # Always update the last checkpoint
        checkpoint_last = f"{cfg['to_save_checkpoint']}_last.pth"
        torch.save(checkpoint_data, checkpoint_last)
        print(f"Last checkpoint saved at {checkpoint_last}")
        
        # Save best checkpoint if validation loss improved
        if val_loss < early_stopper.best_loss:
            checkpoint_best = f"{cfg['to_save_checkpoint']}_best.pth"
            torch.save(checkpoint_data, checkpoint_best)
            print(f"Best checkpoint saved at {checkpoint_best}")

    else:
        # Assume save_method is a number indicating how many checkpoints to save
        try:
            save_number = int(save_method)
        except ValueError:
            print("Invalid save_method value in config, defaulting to saving all checkpoints.")
            save_number = None

        if save_number is not None:
            num_epochs = cfg["epochs"]
            checkpoints_epochs = set(round(i * num_epochs / save_number) for i in range(1, save_number + 1))
            if epoch in checkpoints_epochs:
                checkpoint_path = f"{cfg['to_save_checkpoint']}_{epoch}.pth"
                torch.save(checkpoint_data, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

    # Check early stopping
    if early_stopper.step(val_loss):  
        print(f"Early stopping at epoch {epoch}")
        return True  # Signal to stop training

    return False  # Continue training


def validate(face_model, generator, val_loader, device):
    """Runs validation on the dataset."""
    face_model.eval()
    generator.eval()
    
    total_embedding_loss, total_gen_loss = 0, 0
    num_batches = 0

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", unit="batch") as pbar:
            for mini_batches in val_loader:
                for batch in mini_batches:
                    batch = {key: [b[key] for b in batch] 
                             if key.endswith('id') 
                             else torch.stack([b[key] for b in batch]).to(device)
                             for key in batch[0].keys()}

                    images = batch['tensor'].to(device)
                    embeddings = face_model(images)

                    mean_embedding = embeddings.mean(dim=0, keepdim=True)
                    variance_loss = ((embeddings - mean_embedding) ** 2).mean()

                    generated_images_for_face = generator(embeddings).detach()
                    mse_loss_face = nn.MSELoss()(generated_images_for_face, images)

                    embedding_loss = variance_loss + mse_loss_face

                    embeddings_for_gen = face_model(images).detach()
                    generated_images = generator(embeddings_for_gen)
                    mse_loss_gen = nn.MSELoss()(generated_images, images)

                    total_embedding_loss += embedding_loss.item()
                    total_gen_loss += mse_loss_gen.item()
                    num_batches += 1

                    pbar.set_postfix({
                        "Emb Loss": f"{embedding_loss:.4f}",
                        "Gen Loss": f"{mse_loss_gen:.4f}"
                    })
                    pbar.update(1)

    avg_embedding_loss = total_embedding_loss / num_batches
    avg_gen_loss = total_gen_loss / num_batches
    return avg_embedding_loss, avg_gen_loss


def train():
    """Full training loop with validation, checkpoint loading, loss history tracking, and early stopping."""
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    face_model = FaceEmbeddingModel(cfg["model_name"], cfg["embedding_size"]).to(device)
    generator = Generator(cfg["embedding_size"]).to(device)

    # Load existing checkpoint if available
    if cfg["existing_checkpoint"] and os.path.exists(cfg["existing_checkpoint"]):
        checkpoint = torch.load(cfg["existing_checkpoint"], map_location=device)
        face_model.load_state_dict(checkpoint["face_model_state"])
        generator.load_state_dict(checkpoint["generator_state"])
        print(f"Loaded checkpoint from {cfg['existing_checkpoint']}")
    else:
        print(f"Checkpoint {cfg['existing_checkpoint']} not found. Starting from scratch.")

    # Optimizers
    optimizer_face = optim.Adam(face_model.parameters(), lr=cfg["lr"])
    optimizer_gen = optim.Adam(generator.parameters(), lr=cfg["lr"])

    # Load dataset & create DataLoader with train/val split
    dataset = CustomImageDataset(cfg)
    train_size = int(cfg['train_size'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    sampler = GroupedSampler(train_dataset, cfg["batch_size"])
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=custom_collate_fn)

    val_sampler = GroupedSampler(val_dataset, cfg["batch_size"])
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=custom_collate_fn)

    num_epochs = cfg["epochs"]

    # Initialize loss history & early stopping
    loss_history = {"train_emb": [], "train_gen": [], "val_emb": [], "val_gen": []}
    early_stopper = EarlyStopping(patience=cfg.get("early_stopping_patience", 5))

    for epoch in range(1, num_epochs + 1):
        face_model.train()
        generator.train()

        total_embedding_loss, total_gen_loss = 0, 0
        num_batches = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for mini_batches in train_loader:
                for batch in mini_batches:
                    batch = {key: [b[key] for b in batch] 
                             if key.endswith('id') 
                             else torch.stack([b[key] for b in batch]).to(device)
                             for key in batch[0].keys()}

                    embedding_loss, gen_loss = train_step(face_model, generator, batch, optimizer_face, optimizer_gen, device)

                    total_embedding_loss += embedding_loss
                    total_gen_loss += gen_loss
                    num_batches += 1

                    pbar.set_postfix({
                        "Emb Loss": f"{embedding_loss:.4f}",
                        "Gen Loss": f"{gen_loss:.4f}"
                    })
                    pbar.update(1)

        avg_embedding_loss = total_embedding_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches

        print(f"\nEpoch {epoch} - Avg Emb Loss: {avg_embedding_loss:.4f}, Avg Gen Loss: {avg_gen_loss:.4f}")

        # Run validation at the end of epoch
        val_embedding_loss, val_gen_loss = validate(face_model, generator, val_loader, device)
        print(f"\nValidation - Emb Loss: {val_embedding_loss:.4f}, Gen Loss: {val_gen_loss:.4f}")

        # Save loss history
        loss_history["train_emb"].append(avg_embedding_loss)
        loss_history["train_gen"].append(avg_gen_loss)
        loss_history["val_emb"].append(val_embedding_loss)
        loss_history["val_gen"].append(val_gen_loss)

        # Save checkpoint
        should_stop = save_checkpoint(cfg, epoch, face_model, generator, optimizer_face, optimizer_gen, loss_history, val_embedding_loss + val_gen_loss, early_stopper)

        if should_stop:
            break  # Stop training if early stopping is triggered


    # Save final loss visualization
    plot_loss(loss_history, save_path="loss_plot.png")


if __name__ == "__main__":
    train()