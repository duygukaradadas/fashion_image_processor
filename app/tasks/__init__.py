"""
Task module containing Celery background tasks.
"""

# Import all tasks for Celery autodiscovery
from app.tasks.ping import ping
from app.tasks.embedding_task import generate_embeddings_task
