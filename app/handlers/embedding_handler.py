import math
import os
from fastapi import Query, BackgroundTasks, Path, HTTPException
from typing import Optional, List
from app.tasks.embedding_task import (
    generate_embeddings_task,
    generate_single_embedding_task,
    update_single_embedding_task,
    delete_single_embedding_task
)
from app.services.api_client import ApiClient


async def generate_embeddings(
    start_page: int = Query(1, description="Başlangıç sayfa numarası"),
    batch_size: int = Query(100, description="Her CSV kaydı için işlenecek sayfa sayısı (varsayılan 100)"),
    output_file: Optional[str] = Query(None, description="Çıktı CSV dosyası yolu")
):
    """
    Tüm ürün sayfalarından embedding oluşturmayı tetikleyen handler.
    İşi CELERY_WORKER_CONCURRENCY ortam değişkenine göre (varsayılan 4) parçaya böler.
    MAX_PAGES_OVERALL environment variable'ı ile toplam işlenecek sayfa sayısı sınırlandırılabilir.
    """
    api_client = ApiClient()

    max_pages_env_str = os.getenv('MAX_PAGES_OVERALL')
    max_pages_to_process_overall: Optional[int] = None
    if max_pages_env_str and max_pages_env_str.isdigit():
        max_pages_to_process_overall = int(max_pages_env_str)
    elif max_pages_env_str:
        print(f"Warning: MAX_PAGES_OVERALL environment variable ('{max_pages_env_str}') is not a valid integer. Processing all pages.")

    celery_concurrency_env_str = os.getenv('CELERY_WORKER_CONCURRENCY')
    num_target_tasks = 4
    if celery_concurrency_env_str and celery_concurrency_env_str.isdigit():
        parsed_concurrency = int(celery_concurrency_env_str)
        if parsed_concurrency > 0:
            num_target_tasks = parsed_concurrency
        else:
            print(f"Warning: CELERY_WORKER_CONCURRENCY ('{celery_concurrency_env_str}') must be positive. Defaulting to {num_target_tasks} tasks.")
    elif celery_concurrency_env_str:
        print(f"Warning: CELERY_WORKER_CONCURRENCY ('{celery_concurrency_env_str}') is not a valid integer. Defaulting to {num_target_tasks} tasks.")

    try:
        first_page_data = await api_client.get_products(page=1)
        if not first_page_data.meta:
            raise HTTPException(status_code=500, detail="API'den sayfa meta verisi alınamadı.")
        
        total_api_pages = first_page_data.meta.last_page
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"API'ye bağlanırken veya ilk sayfa alınırken hata: {str(e)}")

    effective_start_page = start_page
    
    if effective_start_page > total_api_pages:
        return {
            "message": f"Başlangıç sayfası ({effective_start_page}) toplam API sayfasından ({total_api_pages}) büyük. İşlem yapılmadı.",
            "task_ids": []
        }

    pages_from_start = total_api_pages - effective_start_page + 1

    if max_pages_to_process_overall is not None:
        num_pages_to_actually_process = min(pages_from_start, max_pages_to_process_overall)
    else:
        num_pages_to_actually_process = pages_from_start
        
    if num_pages_to_actually_process <= 0:
        return {
            "message": "İşlenecek sayfa sayısı 0 veya daha az. İşlem yapılmadı.",
            "task_ids": []
        }

    actual_num_dispatch_tasks = min(num_target_tasks, num_pages_to_actually_process)
    if actual_num_dispatch_tasks <= 0 and num_pages_to_actually_process > 0:
        actual_num_dispatch_tasks = 1

    pages_per_task = math.ceil(num_pages_to_actually_process / actual_num_dispatch_tasks)
    
    task_ids = []
    
    for i in range(actual_num_dispatch_tasks):
        chunk_start_page = effective_start_page + (i * pages_per_task)
        
        if chunk_start_page > total_api_pages:
            break

        num_pages_for_this_chunk = min(pages_per_task, total_api_pages - chunk_start_page + 1)
        
        processed_so_far_not_including_this_chunk = i * pages_per_task
        remaining_pages_to_assign = num_pages_to_actually_process - processed_so_far_not_including_this_chunk
        num_pages_for_this_chunk = min(num_pages_for_this_chunk, remaining_pages_to_assign)

        if num_pages_for_this_chunk <= 0:
            continue

        task = generate_embeddings_task.delay(
            start_page=chunk_start_page,
            max_pages=num_pages_for_this_chunk,
            batch_size=batch_size,
            output_file=output_file
        )
        task_ids.append(task.id)
        
    return {
        "message": f"{len(task_ids)} adet embedding oluşturma görevi {num_pages_to_actually_process} sayfa için başlatıldı (her biri yaklaşık {pages_per_task} sayfa).",
        "task_ids": task_ids,
        "dispatch_details": {
            "target_tasks_based_on_env_CELERY_WORKER_CONCURRENCY": num_target_tasks,
            "actual_tasks_dispatched": actual_num_dispatch_tasks,
            "pages_per_task_approx": pages_per_task
        },
        "request_params_used": {
             "start_page": start_page,
             "batch_size": batch_size,
             "output_file": output_file,
             "max_pages_to_process_overall_env": max_pages_to_process_overall if max_pages_to_process_overall is not None else "all (env not set or invalid)"
        }
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
