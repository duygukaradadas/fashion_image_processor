# Fashion Image Processor

> **Note:** All endpoints are now public and do not require authentication.

A microservice for generating and managing image embeddings for fashion products using ResNet50. The service provides similarity search capabilities based on visual features extracted from product images.

## Features

- Generate embeddings from fashion product images using ResNet50
- Store embeddings in FAISS for fast similarity search
- RESTful API for similarity search
- Asynchronous processing with Celery
- Docker support for easy deployment

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
curl http://localhost:8000/embeddings/check/20
```

2. Get list of products with embeddings:
```bash
curl http://localhost:8000/embeddings/products
```

3. Get similar products with adjusted threshold:
```bash
curl "http://localhost:8000/similar-products/20?top_n=10&score_threshold=0.001"
```

4. Check total number of embeddings:
```bash
curl http://localhost:8000/embeddings/count
```

5. Generate embedding for a single product:
```bash
curl -X POST http://localhost:8000/embeddings/product/20
```

6. Update embedding for a product:
```bash
curl -X PUT http://localhost:8000/embeddings/product/20
```

7. Delete embedding for a product:
```bash
curl -X DELETE http://localhost:8000/embeddings/product/20
```

8. Generate embeddings for multiple products:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"product_ids": [20, 29, 14]}' http://localhost:8000/embeddings/generate-batch
```

9. Generate embeddings for all products (paginated):
```bash
curl -X POST http://localhost:8000/embeddings/generate-all?start_page=1&batch_size=10
```

### Find Similar Products by Image Upload

**POST** `/similar-products/upload`

- **Description:** Upload an image to get similar products.
- **Request:**  
  - Form-data with key `image` (the image file)
  - Optional query params: `top_n`, `score_threshold`
- **Example:**
  ```bash
  curl -X POST -F "image=@/path/to/image.jpg" "http://localhost:8000/similar-products/upload?top_n=5&score_threshold=0.001"
  ```
- **Response:**
  ```json
  {
    "similar_products": [
      {
        "id": 23,
        "similarity_score": 1.0,
        "image_url": "https://fashion.aknevrnky.dev/api/products/23/image"
      },
      ...
    ]
  }
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

- All endpoints are now public and do not require authentication.
- Similarity search uses L2 distance, where lower values indicate higher similarity
- The recommended similarity threshold is 0.001
- Batch operations are processed asynchronously using Celery 