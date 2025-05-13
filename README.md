# Fashion Image Processor

A microservice for generating and managing image embeddings for fashion products using ResNet50. The service provides similarity search capabilities based on visual features extracted from product images.

## Features

- Generate embeddings from fashion product images using ResNet50
- Store embeddings in Redis for fast similarity search
- RESTful API for embedding generation and similarity search
- Asynchronous processing with Celery
- Docker support for easy deployment

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Redis

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd fashion-image-processor
```

2. Create required directories:
```bash
mkdir embeddings
```

3. Start the services using Docker Compose:
```bash
docker compose up -d
```

This will start:
- FastAPI application on port 4000
- Redis on port 6379
- Celery worker for background tasks

## API Endpoints

### Generate Embeddings
```http
POST /embeddings/generate
```
Generates embeddings for all products in the database.

### Find Similar Products
```http
GET /similar-products/{product_id}?top_n=5
```
Returns the most similar products based on image embeddings.

### Health Check
```http
GET /ping
```
Returns the service status.

## Development

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Celery worker:
```bash
celery -A app.celery_config.celery_app worker --loglevel=info
```

## Architecture

The service consists of several components:
- FastAPI application for HTTP endpoints
- ResNet50 model for feature extraction
- Redis for embedding storage
- Celery for background task processing

## Configuration

Environment variables:
- `API_BASE_URL`: Base URL for the fashion API
- `REDIS_URL`: Redis connection URL
- `CELERY_BROKER_URL`: Celery broker URL

## License

MIT License

# Create the client
client = ApiClient(base_url="https://fashion.aknevrnky.dev")

# Get image bytes by product ID
image_data = await client.get_product_image(product_id=1)

# Or directly from a URL
image_data = await client.get_product_image(image_url="https://fashion.aknevrnky.dev/storage/products/10000.jpg")

# Process with PIL/Pillow
from PIL import Image
import io
image = Image.open(io.BytesIO(image_data))
# Now the image can be processed for training