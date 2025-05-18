import os
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any
from app.services.faiss_service import FAISSService
from app.resnet.model import ResNetEmbedder
from app.services.api_client import ApiClient
import asyncio
from PIL import Image
import io

class EmbeddingService:
    def __init__(self, api_base_url: Optional[str] = None):
        self.faiss_service = FAISSService()
        self.model = ResNetEmbedder()
        self.api_client = ApiClient(base_url=api_base_url)
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """Get embedding for a product from FAISS."""
        return self.faiss_service.get_embedding(product_id)
    
    def save_embedding(self, product_id: int, embedding: np.ndarray) -> bool:
        """Save embedding to FAISS."""
        if not isinstance(embedding, np.ndarray) or embedding.shape != (2048,):
            raise ValueError("Invalid embedding format or size")
        return self.faiss_service.save_embedding(product_id, embedding.tolist())
    
    def delete_embedding(self, product_id: int) -> bool:
        """Delete embedding from FAISS."""
        return self.faiss_service.delete_embedding(product_id)
    
    async def generate_embedding(self, product_id: int) -> Tuple[int, Optional[np.ndarray]]:
        """
        Generate embedding for a product using its image.
        
        Args:
            product_id: Product ID to generate embedding for
            
        Returns:
            Tuple of (product_id, embedding) where embedding is None if generation failed
        """
        try:
            # Get the product image
            image_data = await self.api_client.get_product_image(product_id=product_id)
            
            # Validate image data
            if not image_data:
                print(f"No image data received for product {product_id}")
                return product_id, None
                
            # Convert bytes to PIL Image
            try:
                image = Image.open(io.BytesIO(image_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                print(f"Error processing image for product {product_id}: {str(e)}")
                return product_id, None
            
            # Generate embedding using ResNet model
            embedding = self.model.get_embedding(image)
            
            # Validate embedding
            if embedding is None or embedding.shape != (2048,):
                print(f"Invalid embedding generated for product {product_id}")
                return product_id, None
            
            # Save to FAISS
            if self.faiss_service.save_embedding(product_id, embedding.tolist()):
                return product_id, embedding
            else:
                print(f"Failed to save embedding for product {product_id}")
                return product_id, None
                
        except Exception as e:
            print(f"Error generating embedding for product {product_id}: {str(e)}")
            return product_id, None
    
    async def generate_embeddings_batch(self, product_ids: List[int], batch_size: int = 10) -> Dict[int, Optional[np.ndarray]]:
        """
        Generate embeddings for multiple products in batches.
        
        Args:
            product_ids: List of product IDs to generate embeddings for
            batch_size: Number of products to process in parallel
            
        Returns:
            Dictionary mapping product IDs to their embeddings (None if generation failed)
        """
        results = {}
        
        # Process products in batches
        for i in range(0, len(product_ids), batch_size):
            batch = product_ids[i:i + batch_size]
            
            # Generate embeddings for batch in parallel
            tasks = [self.generate_embedding(pid) for pid in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Update results dictionary
            for pid, embedding in batch_results:
                results[pid] = embedding
            
            # Small delay between batches to avoid overwhelming the API
            if i + batch_size < len(product_ids):
                await asyncio.sleep(0.5)
        
        return results
    
    def find_similar_products(self, embedding: Union[List[float], np.ndarray], top_n: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar products using vector similarity search.
        
        Args:
            embedding: Query embedding vector
            top_n: Number of similar products to return
            score_threshold: Minimum similarity score threshold (0-1)
            
        Returns:
            List of dictionaries containing product IDs and similarity scores
        """
        return self.faiss_service.find_similar_products(embedding, top_n, score_threshold)
    
    def find_similar_by_id(self, product_id: int, top_n: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar products using a product ID as reference.
        
        Args:
            product_id: Reference product ID
            top_n: Number of similar products to return
            score_threshold: Minimum similarity score threshold (0-1)
            
        Returns:
            List of dictionaries containing product IDs and similarity scores
        """
        return self.faiss_service.find_similar_by_id(product_id, top_n, score_threshold)