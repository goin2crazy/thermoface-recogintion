import torch
import torch.nn as nn
import timm
from config import load_config


class FaceEmbeddingModel(nn.Module):
    """
    Model that takes an image and outputs a 1024-dimensional embedding.
    """

    def __init__(self, model_name, embedding_size):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # Remove classifier
        self.fc = nn.Linear(self.backbone.num_features, embedding_size)  # Add final embedding layer

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        return embedding  # Output shape: [batch_size, 1024]

class Generator(nn.Module):
    """
    Generator model that reconstructs an image from an embedding.
    """

    def __init__(self, embedding_size, img_size=(3, 224, 224)):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(embedding_size, 512 * 7 * 7)  # Upscale to small feature map
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.Tanh(),  # Output range [-1, 1]
        )

    def forward(self, embedding):
        x = self.fc(embedding)  # Shape: [batch_size, 512*7*7]
        x = x.view(-1, 512, 7, 7)  # Reshape to 7x7 feature map
        x = self.deconv(x)  # Upscale to [batch_size, 3, 224, 224]
        return x

if __name__ == "__main__": 
    # Load config
    cfg = load_config()
    model_name = cfg["model_name"]  # Should be "resnet50d"
    embedding_size = cfg["embedding_size"]  # Should be 1024

    # Example usage
    face_model = FaceEmbeddingModel(model_name, embedding_size)
    generator = Generator(embedding_size)

    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    embedding = face_model(dummy_input)
    reconstructed_images = generator(embedding)

    print(f"Embedding shape: {embedding.shape}")  # Should be [2, 1024]
    print(f"Reconstructed image shape: {reconstructed_images.shape}")  # Should be [2, 3, 224, 224]
