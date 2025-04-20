import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import asyncio
from typing import List, Dict, Optional, Tuple

from app.resnet.model import ResNetEmbedder
from app.services.api_client import ApiClient


class EmbeddingService:
    """
    Service for generating and managing embeddings using ResNet50.
    This service integrates the ResNet model with the API client to 
    generate embeddings from product images.
    """
    
    def __init__(self, api_base_url: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            api_base_url: Base URL for the API client
        """
        self.api_client = ApiClient(base_url=api_base_url)
        self.model = ResNetEmbedder()
        
    async def get_embedding_for_product(self, product_id: int) -> Tuple[int, np.ndarray]:
        """
        Generate embedding for a single product.
        
        Args:
            product_id: ID of the product
            
        Returns:
            Tuple containing (product_id, embedding vector)
            
        Raises:
            Exception: If image retrieval or embedding generation fails
        """
        # Get the product image
        image_data = await self.api_client.get_product_image(product_id=product_id)
        
        # Generate embedding
        embedding = self.model.get_embedding(image_data)
        
        return (product_id, embedding)
    
    async def generate_embeddings_for_products(
        self, 
        product_ids: List[int], 
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate embeddings for multiple products.
        
        Args:
            product_ids: List of product IDs
            output_file: Optional CSV file to save embeddings to
            
        Returns:
            DataFrame containing product IDs and their embeddings
            
        Raises:
            Exception: If any part of the embedding generation process fails
        """
        embeddings = []
        ids = []
        
        # Use tqdm to show progress
        for product_id in tqdm(product_ids, desc="Generating embeddings"):
            try:
                product_id, embedding = await self.get_embedding_for_product(product_id)
                embeddings.append(embedding)
                ids.append(product_id)
            except Exception as e:
                print(f"Error processing product {product_id}: {str(e)}")
        
        # Create DataFrame
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.insert(0, "id", ids)
        
        # Save to CSV if output_file is provided
        if output_file:
            embeddings_df.to_csv(output_file, index=False)
            print(f"Embeddings saved to {output_file}")
        
        return embeddings_df
    
    async def generate_embeddings_from_page(
        self, 
        page: int = 1, 
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate embeddings for products from a specific page in the API.
        
        Args:
            page: Page number to fetch products from
            output_file: Optional CSV file to save embeddings to
            
        Returns:
            DataFrame containing product IDs and their embeddings
            
        Raises:
            Exception: If API request or embedding generation fails
        """
        # Get products from the API
        products_response = await self.api_client.get_products(page=page)
        
        # Extract product IDs
        product_ids = [product.id for product in products_response.data]
        
        # Generate embeddings
        return await self.generate_embeddings_for_products(product_ids, output_file)
    
    async def generate_embeddings_for_all_products(
        self, 
        start_page: int = 1,
        max_pages: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate embeddings for all products in the API.
        
        Args:
            start_page: Page to start from
            max_pages: Maximum number of pages to process (None = all)
            output_file: Optional CSV file to save embeddings to
            
        Returns:
            DataFrame containing product IDs and their embeddings
            
        Raises:
            Exception: If API request or embedding generation fails
        """
        all_embeddings = []
        all_ids = []
        
        # Start with the first page
        page = start_page
        pages_processed = 0
        
        while True:
            # Check if we've reached the maximum number of pages
            if max_pages is not None and pages_processed >= max_pages:
                break
                
            try:
                # Get products from the current page
                products_response = await self.api_client.get_products(page=page)
                
                if not products_response.data:
                    # No more products
                    break
                
                # Extract product IDs
                product_ids = [product.id for product in products_response.data]
                
                # Generate embeddings for this page
                for product_id in tqdm(product_ids, desc=f"Processing page {page}"):
                    try:
                        pid, embedding = await self.get_embedding_for_product(product_id)
                        all_embeddings.append(embedding)
                        all_ids.append(pid)
                    except Exception as e:
                        print(f"Error processing product {product_id}: {str(e)}")
                
                # Check if there's a next page
                if not products_response.links.next:
                    # No more pages
                    break
                
                # Move to the next page
                page += 1
                pages_processed += 1
                
            except Exception as e:
                print(f"Error processing page {page}: {str(e)}")
                break
        
        # Create DataFrame
        embeddings_df = pd.DataFrame(all_embeddings)
        embeddings_df.insert(0, "id", all_ids)
        
        # Save to CSV if output_file is provided
        if output_file:
            embeddings_df.to_csv(output_file, index=False)
            print(f"Embeddings saved to {output_file}")
        
        return embeddings_df
