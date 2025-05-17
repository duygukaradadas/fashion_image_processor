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
    broker_connection_retry_on_startup=True
)

# Updated task routes to use new task names
celery_app.conf.task_routes = {
    "generate_embedding": {"queue": "celery"},
    "generate_embeddings_batch": {"queue": "celery"},
    "delete_embedding": {"queue": "celery"},
    "app.tasks.ping.ping": {"queue": "celery"},
}
