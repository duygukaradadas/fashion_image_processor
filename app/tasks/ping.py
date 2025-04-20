from app.celery_config import celery_app

@celery_app.task
def ping():
    return "pong"