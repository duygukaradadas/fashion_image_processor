import os
import numpy as np
from typing import Optional, List, Dict, Tuple
from app.services.qdrant_service import QdrantService
from app.resnet.model import ResNetEmbedder
from app.services.api_client import ApiClient

class EmbeddingService:
    def __init__(self, api_base_url: Optional[str] = None):
        self.qdrant_service = QdrantService()
        self.model = ResNetEmbedder()
        self.api_client = ApiClient(base_url=api_base_url)
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """Get embedding for a product from Qdrant."""
        return self.qdrant_service.get_embedding(product_id)
    
    def save_embedding(self, product_id: int, embedding: np.ndarray) -> bool:
        """Save embedding to Qdrant."""
        return self.qdrant_service.save_embedding(product_id, embedding.tolist())
    
    def delete_embedding(self, product_id: int) -> bool:
        """Delete embedding from Qdrant."""
        return self.qdrant_service.delete_embedding(product_id)
    
    async def generate_embedding(self, product_id: int) -> Tuple[int, Optional[np.ndarray]]:
        """Generate embedding for a product using its image."""
        try:
            # Get the product image
            image_data = await self.api_client.get_product_image(product_id=product_id)
            
            # Generate embedding using ResNet model
            embedding = self.model.get_embedding(image_data)
            
            # Save to Qdrant
            self.qdrant_service.save_embedding(product_id, embedding.tolist())
            
            return product_id, embedding
        except Exception as e:
            print(f"Error generating embedding for product {product_id}: {str(e)}")
            return product_id, None
    
    def get_embeddings_batch(self, product_ids: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """Get multiple embeddings from Qdrant."""
        return self.qdrant_service.get_embeddings_batch(product_ids)
    
    def save_embeddings_batch(self, product_ids: List[int], embeddings: np.ndarray) -> bool:
        """Save multiple embeddings to Qdrant."""
        return self.qdrant_service.save_embeddings_batch(product_ids, embeddings)