import os

from celery import Celery

celery_app = Celery(
    "image_processor",
    broker=os.getenv("CELERY_BROKER_DSN"),
    backend=os.getenv("CELERY_BACKEND_DSN"),
)

celery_app.conf.task_routes = {
    "app.tasks.*": {"queue": "default"},
}
