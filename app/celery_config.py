import os

from celery import Celery

# Redis connection settings
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = os.getenv('REDIS_PORT', 6379)

# Celery configuration
celery_app = Celery(
    'fashion_image_processor',
    broker=f'redis://{redis_host}:{redis_port}/0',
    backend=f'redis://{redis_host}:{redis_port}/0',
    include=['app.tasks.embedding_task']
)

# Optional configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    
    # Performance optimizations
    worker_prefetch_multiplier=1,  # Disable prefetching to prevent worker from getting too many tasks
    worker_max_tasks_per_child=1000,  # Restart worker after processing 1000 tasks to prevent memory leaks
    task_time_limit=300,  # 5 minutes timeout for tasks
    task_soft_time_limit=240,  # 4 minutes soft timeout
    worker_concurrency=16,  # Increase number of worker processes
    
    # Task routing
    task_routes={
        "generate_embedding": {"queue": "celery"},
        "generate_embeddings_batch": {"queue": "celery"},
        "delete_embedding": {"queue": "celery"},
        "app.tasks.ping.ping": {"queue": "celery"},
    },
    
    # Task retry settings
    task_acks_late=True,  # Only acknowledge task after it's completed
    task_reject_on_worker_lost=True,  # Requeue task if worker dies
    task_default_retry_delay=30,  # Wait 30 seconds before retrying
    task_max_retries=3,  # Maximum number of retries
)
