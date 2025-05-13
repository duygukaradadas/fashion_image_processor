import os
import asyncio
from celery import shared_task

from app.services.embedding_service import EmbeddingService
from app.celery_config import celery_app


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
