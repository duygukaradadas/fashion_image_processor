from fastapi import Query, BackgroundTasks, Path
from typing import Optional
from app.tasks.embedding_task import (
    generate_embeddings_task,
    generate_single_embedding_task,
    update_single_embedding_task,
    delete_single_embedding_task
)


async def generate_embeddings(
    background_tasks: BackgroundTasks,
    start_page: int = Query(1, description="Başlangıç sayfa numarası"),
    max_pages: Optional[int] = Query(None, description="İşlenecek maksimum sayfa sayısı (None = tümü)"),
    batch_size: int = Query(1, description="Her CSV kaydı için işlenecek sayfa sayısı"),
    output_file: Optional[str] = Query(None, description="Çıktı CSV dosyası yolu")
):
    """
    Tüm ürün sayfalarından embedding oluşturmayı tetikleyen handler.
    Görevi Celery'ye gönderir ve arka planda çalıştırır.
    
    Args:
        background_tasks: FastAPI background tasks
        start_page: Başlangıç sayfa numarası
        max_pages: İşlenecek maksimum sayfa sayısı (None = tümü)
        batch_size: Her CSV kaydı için işlenecek sayfa sayısı
        output_file: Çıktı CSV dosyası yolu
        
    Returns:
        dict: Görev bilgileri
    """
    # Celery görevini sıraya al
    task = generate_embeddings_task.delay(
        start_page=start_page,
        max_pages=max_pages,
        batch_size=batch_size,
        output_file=output_file
    )
    
    return {
        "task_id": task.id,
        "status": "queued",
        "parameters": {
            "start_page": start_page,
            "max_pages": max_pages if max_pages is not None else "all",
            "batch_size": batch_size,
            "output_file": output_file
        },
        "message": "Ürün embeddingi oluşturma görevi başlatıldı. Tüm sayfalar işlenecek ve sonuçlar CSV dosyasına kaydedilecek."
    }


async def generate_single_embedding(
    product_id: int = Path(..., description="Embedding oluşturulacak ürün ID'si")
):
    """
    Belirli bir ürün için embedding oluşturmayı tetikleyen handler.
    
    Args:
        product_id: Embedding oluşturulacak ürün ID'si
        
    Returns:
        dict: Görev bilgileri
    """
    # Celery görevini sıraya al
    task = generate_single_embedding_task.delay(product_id=product_id)
    
    return {
        "task_id": task.id,
        "status": "queued",
        "product_id": product_id,
        "message": f"Ürün ID {product_id} için embedding oluşturma görevi başlatıldı."
    }


async def update_single_embedding(
    product_id: int = Path(..., description="Embedding'i güncellenecek ürün ID'si")
):
    """
    Belirli bir ürün için embedding'i güncellemeyi tetikleyen handler.
    Varolan embedding'i siler ve yeniden oluşturur.
    
    Args:
        product_id: Embedding'i güncellenecek ürün ID'si
        
    Returns:
        dict: Görev bilgileri
    """
    # Celery görevini sıraya al
    task = update_single_embedding_task.delay(product_id=product_id)
    
    return {
        "task_id": task.id,
        "status": "queued",
        "product_id": product_id,
        "message": f"Ürün ID {product_id} için embedding güncelleme görevi başlatıldı."
    }


async def delete_single_embedding(
    product_id: int = Path(..., description="Embedding'i silinecek ürün ID'si")
):
    """
    Belirli bir ürün için embedding'i silmeyi tetikleyen handler.
    
    Args:
        product_id: Embedding'i silinecek ürün ID'si
        
    Returns:
        dict: Görev bilgileri
    """
    # Celery görevini sıraya al
    task = delete_single_embedding_task.delay(product_id=product_id)
    
    return {
        "task_id": task.id,
        "status": "queued",
        "product_id": product_id,
        "message": f"Ürün ID {product_id} için embedding silme görevi başlatıldı."
    }
