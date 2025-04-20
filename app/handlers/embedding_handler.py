from fastapi import Query, BackgroundTasks
from typing import Optional
from app.tasks.embedding_task import generate_embeddings_task


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
