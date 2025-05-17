import os

from celery import Celery

# Default DSNs if not provided by environment variables
DEFAULT_REDIS_DSN = 'redis://redis:6379/0'

# Celery configuration using DSNs
broker_dsn = os.getenv('CELERY_BROKER_DSN', DEFAULT_REDIS_DSN)
backend_dsn = os.getenv('CELERY_BACKEND_DSN', broker_dsn) # Default backend to broker DSN

celery_app = Celery(
    'fashion_processor',
    broker=broker_dsn,
    backend=backend_dsn
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
