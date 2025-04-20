from fastapi import Query, BackgroundTasks
from typing import Optional
from app.tasks.embedding_task import generate_embeddings_task


async def generate_embeddings(
    background_tasks: BackgroundTasks,
    page: int = Query(1, description="Starting page number"),
    max_pages: int = Query(1, description="Maximum number of pages to process"),
    output_file: Optional[str] = Query(None, description="Output CSV file path")
):
    """
    Handler to trigger embedding generation.
    
    Args:
        background_tasks: FastAPI background tasks
        page: Starting page number
        max_pages: Maximum number of pages to process
        output_file: Output CSV file path
        
    Returns:
        dict: Task information
    """
    # Queue the task in Celery
    task = generate_embeddings_task.delay(
        page=page,
        max_pages=max_pages,
        output_file=output_file
    )
    
    return {
        "task_id": task.id,
        "status": "queued",
        "parameters": {
            "page": page,
            "max_pages": max_pages,
            "output_file": output_file
        }
    }
