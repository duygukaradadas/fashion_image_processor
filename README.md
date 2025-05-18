# Fashion Image Processor

A microservice for generating and managing image embeddings for fashion products using ResNet50. The service provides similarity search capabilities based on visual features extracted from product images.

## Features

- Generate embeddings from fashion product images using ResNet50
- Store embeddings in FAISS for fast similarity search
- RESTful API for similarity search
- Asynchronous processing with Celery
- Docker support for easy deployment

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Redis (for Celery)

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
- FastAPI application on port 8000
- Redis on port 6379
- Celery worker for background tasks

## Working with Similarity Search

### Understanding Similarity Scores
- The system uses L2 distance for similarity, where lower values indicate higher similarity
- Similarity scores are typically very small (around 0.004)
- The default threshold of 0.1 was too high, filtering out all similar products
- We found that a threshold of 0.001 works well for finding similar products

### Verified Working Endpoints

1. Check if a product has embeddings:
```bash
curl -H "Authorization: Bearer fashion-api-token-2025" http://localhost:8000/embeddings/check/20
```

2. Get list of products with embeddings:
```bash
curl -H "Authorization: Bearer fashion-api-token-2025" http://localhost:8000/embeddings/products
```

3. Get similar products with adjusted threshold:
```bash
curl -H "Authorization: Bearer fashion-api-token-2025" "http://localhost:8000/similar-products/20?top_n=10&score_threshold=0.001"
```

4. Check total number of embeddings:
```bash
curl -H "Authorization: Bearer fashion-api-token-2025" http://localhost:8000/embeddings/count
```

5. Generate embedding for a single product:
```bash
curl -X POST -H "Authorization: Bearer fashion-api-token-2025" "http://localhost:8000/embeddings/generate/20"
```

6. Update embedding for a product:
```bash
curl -X PUT -H "Authorization: Bearer fashion-api-token-2025" "http://localhost:8000/embeddings/update/20"
```

7. Delete embedding for a product:
```bash
curl -X DELETE -H "Authorization: Bearer fashion-api-token-2025" "http://localhost:8000/embeddings/delete/20"
```

8. Generate embeddings for multiple products:
```bash
curl -X POST -H "Authorization: Bearer fashion-api-token-2025" -H "Content-Type: application/json" -d '{"product_ids": [20, 29, 14]}' "http://localhost:8000/embeddings/generate-batch"
```

9. Generate embeddings for all products (paginated):
```bash
curl -X POST -H "Authorization: Bearer fashion-api-token-2025" "http://localhost:8000/embeddings/generate-all?start_page=1&batch_size=10"
```

### Example Responses

1. Similar Products Response:
```json
{
  "similar_products": [
    {"id": 20, "similarity_score": 1.0, "image_url": "https://fashion.aknevrnky.dev/api/products/20/image"},
    {"id": 29, "similarity_score": 0.00455, "image_url": "https://fashion.aknevrnky.dev/api/products/29/image"},
    {"id": 14, "similarity_score": 0.00440, "image_url": "https://fashion.aknevrnky.dev/api/products/14/image"}
  ]
}
```

2. Task Response (for POST/PUT/DELETE operations):
```json
{
  "task_id": "task-uuid-here",
  "status": "queued"
}
```

3. Products with Embeddings Response:
```json
{
  "products": [1, 2, 3, 4, 5]
}
```

4. Embedding Count Response:
```json
{
  "count": 5
}
```

### Notes on Endpoints

- All endpoints require authentication using the `fashion-api-token-2025` token
- Similarity search uses L2 distance, where lower values indicate higher similarity
- The recommended similarity threshold is 0.001
- Batch operations are processed asynchronously using Celery
- The generate-all endpoint processes products in batches to manage memory usage

## Migration from Qdrant to FAISS

We migrated from Qdrant to FAISS for the following reasons:
1. Simpler deployment (no need for a separate vector database)
2. Better performance for our use case
3. Easier integration with Python
4. Lower resource requirements

The migration process:
1. Removed Qdrant dependencies
2. Implemented FAISS service for similarity search
3. Updated the API endpoints to use FAISS
4. Verified the results with the commands above

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
- FAISS for fast vector similarity search
- Redis for Celery task queue
- Celery for background task processing

## Configuration

Environment variables:
- `API_BASE_URL`: Base URL for the fashion API
- `REDIS_HOST`: Redis host
- `REDIS_PORT`: Redis port
- `API_AUTH_TOKEN`: Authentication token for the fashion API

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