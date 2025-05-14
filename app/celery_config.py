import os

from celery import Celery

# Redis connection settings
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = os.getenv('REDIS_PORT', 6379)

# Celery configuration
celery_app = Celery(
    'fashion_processor',
    broker=f'redis://{redis_host}:{redis_port}/0',
    backend=f'redis://{redis_host}:{redis_port}/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Updated task routes to use simple names
celery_app.conf.task_routes = {
    "generate_embeddings": {"queue": "celery"},
    "generate_single_embedding": {"queue": "celery"},
    "update_single_embedding": {"queue": "celery"},
    "delete_single_embedding": {"queue": "celery"},
    "app.tasks.ping.ping": {"queue": "celery"},
}
