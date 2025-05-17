import asyncio
from celery import shared_task
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
from typing import List, Dict, Optional
import numpy as np

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
        return {
            "product_id": product_id,
            "success": False,
            "error": str(e)
        }

@shared_task
def generate_embeddings_batch(product_ids: List[int]) -> List[Dict]:
    """Generate embeddings for multiple products."""
    results = []
    embedding_service = EmbeddingService()
    
    # Get the current event loop or create a new one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        # Process products sequentially to avoid overwhelming the API
        for product_id in product_ids:
            try:
                # Run the async operation
                product_id, embedding = loop.run_until_complete(
                    embedding_service.generate_embedding(product_id)
                )
                
                if embedding is not None:
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
            except Exception as e:
                results.append({
                    "product_id": product_id,
                    "success": False,
                    "error": str(e)
                })
    finally:
        if loop.is_running():
            loop.stop()
    
    return results

@shared_task
def delete_embedding(product_id: int) -> Dict:
    """Delete embedding for a single product from Qdrant."""
    try:
        qdrant_service = QdrantService()
        success = qdrant_service.delete_embedding(product_id)
        return {
            "product_id": product_id,
            "success": success,
            "error": None
        }
    except Exception as e:
        return {
            "product_id": product_id,
            "success": False,
            "error": str(e)
        }
