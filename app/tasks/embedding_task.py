import asyncio
from celery import shared_task
from app.services.faiss_service import FAISSService
from app.services.embedding_service import EmbeddingService
from typing import List, Dict, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@shared_task
def generate_embedding(product_id: int) -> Dict:
    """Generate embedding for a single product."""
    try:
        embedding_service = EmbeddingService()
        
        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            # Run the async operation
            product_id, embedding = loop.run_until_complete(
                embedding_service.generate_embedding(product_id)
            )
            
            if embedding is not None:
                return {
                    "product_id": product_id,
                    "success": True,
                    "error": None
                }
            return {
                "product_id": product_id,
                "success": False,
                "error": "Failed to generate embedding"
            }
        finally:
            if loop.is_running():
                loop.stop()
            
    except Exception as e:
        logger.error(f"Error generating embedding for product {product_id}: {str(e)}")
        return {
            "product_id": product_id,
            "success": False,
            "error": str(e)
        }

@shared_task
def generate_embeddings_batch(product_ids: List[int], batch_size: int = 20) -> List[Dict]:
    """Generate embeddings for multiple products."""
    results = []
    embedding_service = EmbeddingService()
    faiss_service = FAISSService()
    
    # Get the current event loop or create a new one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        # Process products in smaller sub-batches for better memory management
        for i in range(0, len(product_ids), batch_size):
            sub_batch = product_ids[i:i + batch_size]
            logger.info(f"Processing sub-batch {i//batch_size + 1}, products {i+1} to {min(i+batch_size, len(product_ids))}")
            
            # Process sub-batch in parallel using asyncio.gather
            tasks = [embedding_service.generate_embedding(pid) for pid in sub_batch]
            batch_results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            # Collect successful embeddings for batch save
            successful_embeddings = []
            successful_ids = []
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing: {str(result)}")
                    continue
                    
                product_id, embedding = result
                if embedding is not None:
                    successful_embeddings.append(embedding.tolist())
                    successful_ids.append(product_id)
                    results.append({
                        "product_id": product_id,
                        "success": True,
                        "error": None
                    })
                else:
                    results.append({
                        "product_id": product_id,
                        "success": False,
                        "error": "Failed to generate embedding"
                    })
            
            # Save successful embeddings in batch
            if successful_embeddings:
                faiss_service.save_embeddings_batch(successful_ids, successful_embeddings)
            
            # Small delay between sub-batches to prevent overwhelming the system
            if i + batch_size < len(product_ids):
                loop.run_until_complete(asyncio.sleep(0.1))
                
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
    finally:
        if loop.is_running():
            loop.stop()
    
    return results

@shared_task
def delete_embedding(product_id: int) -> Dict:
    """Delete embedding for a single product from FAISS."""
    try:
        faiss_service = FAISSService()
        success = faiss_service.delete_embedding(product_id)
        return {
            "product_id": product_id,
            "success": success,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error deleting embedding for product {product_id}: {str(e)}")
        return {
            "product_id": product_id,
            "success": False,
            "error": str(e)
        }
