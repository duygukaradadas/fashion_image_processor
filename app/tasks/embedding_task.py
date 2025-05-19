import os
import asyncio
from celery import shared_task

from app.services.embedding_service import EmbeddingService
from app.services.redis_service import RedisService
from app.celery_config import celery_app
from app.services.qdrant_service import QdrantService

@celery_app.task(name="generate_embeddings")
def generate_embeddings_task(start_page=1, max_pages=None, output_file=None, batch_size=1):
    """
    Tüm ürün sayfalarından embedding oluşturan Celery görevi.
    
    Args:
        start_page: Başlangıç sayfa numarası
        max_pages: İşlenecek maksimum sayfa sayısı (None = tümü)
        output_file: Çıktı CSV dosyası yolu
        batch_size: Her kaydedilmeden önce işlenecek sayfa grubu boyutu
        
    Returns:
        dict: Görev sonuç bilgileri
    """
    # Varsayılan çıktı dosyası belirtilmemişse oluştur
    if not output_file:
        output_dir = os.path.join(os.getcwd(), "embeddings")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "product_embeddings.csv")
    
    # Embedding servisi oluştur
    service = EmbeddingService()
    
    # Çalıştırılacak asenkron fonksiyonu tanımla
    async def process_all_pages():
        try:
            result = await service.process_all_pages(
                output_file=output_file,
                start_page=start_page,
                max_pages=max_pages,
                batch_size=batch_size
            )
            return result
        except Exception as e:
            print(f"Error in process_all_pages: {str(e)}")
            raise
    
    # Create a new event loop for this task
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_all_pages())
    except Exception as e:
        print(f"Error running async task: {str(e)}")
        raise
    finally:
        loop.close()
    
    return {
        "status": "completed",
        "start_page": start_page,
        "max_pages": max_pages,
        "batch_size": batch_size,
        "output_file": output_file,
        "total_pages_processed": result["total_pages_processed"],
        "total_products_processed": result["total_products_processed"],
        "device_info": result["device_info"]
    }


@celery_app.task(name="generate_single_embedding")
def generate_single_embedding_task(product_id):
    """
    Tek bir ürün için embedding oluşturan Celery görevi.
    
    Args:
        product_id: Embedding oluşturulacak ürün ID'si
        
    Returns:
        dict: Görev sonuç bilgileri
    """
    # Embedding servisi oluştur
    service = EmbeddingService()
    
    # Çalıştırılacak asenkron fonksiyonu tanımla
    async def process_single_product():
        try:
            product_id_result, embedding = await service.get_embedding_for_product(product_id)
            return {
                "product_id": product_id_result,
                "embedding_dimension": embedding.shape[0],
                "success": True
            }
        except Exception as e:
            print(f"Error processing product {product_id}: {str(e)}")
            return {
                "product_id": product_id,
                "success": False,
                "error": str(e)
            }
    
    # Create a new event loop for this task
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_single_product())
    except Exception as e:
        print(f"Error running async task: {str(e)}")
        raise
    finally:
        loop.close()
    
    return {
        "status": "completed" if result.get("success", False) else "failed",
        "product_id": product_id,
        "result": result
    }


@celery_app.task(name="update_single_embedding")
def update_single_embedding_task(product_id):
    """
    Tek bir ürün için embedding'i güncelleyen Celery görevi.
    
    Args:
        product_id: Embedding'i güncellenecek ürün ID'si
        
    Returns:
        dict: Görev sonuç bilgileri
    """
    # Embedding servisi oluştur
    service = EmbeddingService()
    
    # Çalıştırılacak asenkron fonksiyonu tanımla
    async def update_single_product():
        try:
            product_id_result, embedding = await service.update_embedding_for_product(product_id)
            return {
                "product_id": product_id_result,
                "embedding_dimension": embedding.shape[0],
                "success": True
            }
        except Exception as e:
            print(f"Error updating product {product_id}: {str(e)}")
            return {
                "product_id": product_id,
                "success": False,
                "error": str(e)
            }
    
    # Create a new event loop for this task
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(update_single_product())
    except Exception as e:
        print(f"Error running async task: {str(e)}")
        raise
    finally:
        loop.close()
    
    return {
        "status": "completed" if result.get("success", False) else "failed",
        "product_id": product_id,
        "result": result
    }


@celery_app.task(name="delete_single_embedding")
def delete_single_embedding_task(product_id):
    """
    Delete embedding for a single product from both Redis and Qdrant.
    """
    try:
        # Redis servisi oluştur
        redis_service = RedisService()
        # Qdrant servisi oluştur
        qdrant_service = QdrantService()

        # Embedding'i sil
        redis_success = redis_service.delete_embedding(product_id)
        qdrant_success = qdrant_service.delete_embedding(product_id)

        return {
            "status": "completed",
            "product_id": product_id,
            "success": redis_success or qdrant_success
        }
    except Exception as e:
        print(f"Error deleting embedding for product {product_id}: {str(e)}")
        return {
            "status": "failed",
            "product_id": product_id,
            "success": False,
            "error": str(e)
        }
