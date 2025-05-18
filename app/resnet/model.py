import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import os
from pathlib import Path


class ResNetEmbedder:
    """
    ResNet50 model for generating image embeddings.
    This class loads a pre-trained ResNet50 model and provides methods to generate 
    embeddings from images.
    """
    
    _instance = None
    _model = None
    _transform = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResNetEmbedder, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the ResNet50 model and transformation pipeline."""
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final fully connected layer to get embeddings
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 as required by ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    
    def get_embedding(self, image_data):
        """
        Generate an embedding for a single image.
        
        Args:
            image_data: Raw image data as bytes or PIL Image
            
        Returns:
            numpy.ndarray: A 2048-dimensional embedding vector
        """
        try:
            # Convert bytes to PIL Image if needed
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            else:
                image = image_data.convert("RGB")
            
            # Transform the image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(img_tensor).squeeze().cpu().numpy()
                
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def get_device_info(self):
        """
        Get information about the device being used.
        
        Returns:
            dict: Device information
        """
        info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cudnn_enabled": torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
            "cudnn_benchmark": torch.backends.cudnn.benchmark if torch.cuda.is_available() else False
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                "cuda_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
            })
            
        return info
