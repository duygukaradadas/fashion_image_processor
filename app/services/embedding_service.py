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
        output_file: Optional[str] = None,
        append_mode: bool = False
    ) -> pd.DataFrame:
        """
        Generate embeddings for multiple products.
        
        Args:
            product_ids: List of product IDs
            output_file: Optional CSV file to save embeddings to
            append_mode: Whether to append to existing CSV file
            
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
            if append_mode and os.path.exists(output_file):
                # Append without headers if file exists
                embeddings_df.to_csv(output_file, mode='a', header=False, index=False)
                print(f"Appended {len(embeddings_df)} embeddings to {output_file}")
            else:
                # Create new file with headers
                embeddings_df.to_csv(output_file, index=False)
                print(f"Created new embeddings file with {len(embeddings_df)} embeddings at {output_file}")
        
        return embeddings_df
    
    async def process_all_pages(
        self,
        output_file: str,
        start_page: int = 1,
        max_pages: Optional[int] = None,
        batch_size: int = 1
    ) -> Dict:
        """
        Process all available pages of products from the API.
        Each processed page will be saved to the CSV file incrementally.
        
        Args:
            output_file: CSV file to save embeddings to (will be created or appended to)
            start_page: Page number to start from (default: 1)
            max_pages: Maximum number of pages to process (None = all available)
            batch_size: Number of pages to process at once before saving
            
        Returns:
            Dict with processing statistics
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        total_products = 0
        total_pages = 0
        current_page = start_page
        has_more_pages = True
        append_mode = False  # First batch creates the file, subsequent batches append
        
        print(f"Starting embedding generation from page {start_page}, saving to {output_file}")
        
        while has_more_pages:
            if max_pages is not None and total_pages >= max_pages:
                print(f"Reached maximum page limit of {max_pages}")
                break
                
            # Process a batch of pages
            batch_pages_processed = 0
            batch_products_processed = 0
            
            for _ in range(batch_size):
                if max_pages is not None and total_pages >= max_pages:
                    break
                    
                try:
                    print(f"Fetching products from page {current_page}")
                    # Get products from current page
                    products_response = await self.api_client.get_products(page=current_page)
                    
                    if not products_response.data:
                        print("No more products found")
                        has_more_pages = False
                        break
                    
                    # Extract product IDs
                    product_ids = [product.id for product in products_response.data]
                    
                    # Generate and save embeddings for this page
                    await self.generate_embeddings_for_products(
                        product_ids=product_ids,
                        output_file=output_file,
                        append_mode=append_mode
                    )
                    
                    # Update counters
                    batch_pages_processed += 1
                    batch_products_processed += len(product_ids)
                    
                    # Check if there are more pages
                    if not products_response.links.next:
                        print("No more pages available")
                        has_more_pages = False
                        break
                    
                    # Next page for next iteration
                    current_page += 1
                    append_mode = True  # Switch to append mode after first page
                    
                except Exception as e:
                    print(f"Error processing page {current_page}: {str(e)}")
                    # Continue with next page even if there's an error
                    current_page += 1
                    append_mode = True
            
            # Update overall stats
            total_pages += batch_pages_processed
            total_products += batch_products_processed
            
            print(f"Processed batch: {batch_pages_processed} pages, {batch_products_processed} products")
            print(f"Total so far: {total_pages} pages, {total_products} products")
        
        return {
            "total_pages_processed": total_pages,
            "total_products_processed": total_products,
            "output_file": output_file,
            "device_info": self.model.get_device_info()
        }
