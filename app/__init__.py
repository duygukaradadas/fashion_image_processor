from dotenv import load_dotenv

# load .env file
load_dotenv()

# Import celery config to ensure it's loaded
from app.celery_config import celery_app

# Import tasks to ensure they're registered
import app.tasks
