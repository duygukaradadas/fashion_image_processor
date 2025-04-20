import os
import asyncio
from celery import shared_task

from app.services.embedding_service import EmbeddingService
from app.celery_config import celery_app


@celery_app.task(name="generate_embeddings")
def generate_embeddings_task(page=1, max_pages=1, output_file=None):
    """
    Celery task to generate embeddings for products.
    
    Args:
        page: Starting page number
        max_pages: Maximum number of pages to process
        output_file: Output CSV file path
        
    Returns:
        dict: Task result information
    """
    # Set default output file if not provided
    if not output_file:
        output_dir = os.path.join(os.getcwd(), "embeddings")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"embeddings_page_{page}.csv")
    
    # Create embedding service
    service = EmbeddingService(api_base_url='https://fashion.aknevrnky.dev')
    
    # Define async function to run
    async def run_embedding_generation():
        if max_pages == 1:
            # Generate embeddings for a single page
            await service.generate_embeddings_from_page(page=page, output_file=output_file)
        else:
            # Generate embeddings for multiple pages
            await service.generate_embeddings_for_all_products(
                start_page=page,
                max_pages=max_pages,
                output_file=output_file
            )
    
    # Run the async function
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_embedding_generation())
    
    return {
        "status": "completed",
        "start_page": page,
        "max_pages": max_pages,
        "output_file": output_file,
        "device_info": service.model.get_device_info()
    }
