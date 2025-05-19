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

### Generate Embeddings for All Products
```http
POST /embeddings/generate
```
Generates embeddings for all products in the database.

### Generate Embedding for a Specific Product
```http
POST /embeddings/product/{product_id}
```
Generates an embedding only for the product with the given product ID.

### Update Embedding for a Specific Product
```http
PUT /embeddings/product/{product_id}
```
Updates the embedding for the product with the given product ID.

### Delete Embedding for a Specific Product
```http
DELETE /embeddings/product/{product_id}
```
Deletes the embedding for the product with the given product ID.

### Find Similar Products
```http
GET /similar-products/{product_id}?top_n=5
```
Returns the most similar products based on image embeddings.

### Find Similar Products from an Uploaded Image
```http
POST /similar-products/from-image
```
Uploads an image, generates its embedding, and returns the most visually similar products.

**Parameters:**
- `top_n` (form field, optional): Number of similar products to return (default: 5)
- `image` (form field, required): The image file to upload

**Example cURL usage:**
```bash
curl -X POST "http://localhost:4000/similar-products/from-image" \
  -H "Authorization: Bearer <your_token>" \
  -F "image=@/path/to/your/image.jpg" \
  -F "top_n=5"
```

**Response:**
```json
{
  "similar_products": [
    {"id": 23272, "similarity": 0.7769556},
    {"id": 26688, "similarity": 0.7688159},
    {"id": 19482, "similarity": 0.76556134},
    {"id": 24246, "similarity": 0.76446295},
    {"id": 19025, "similarity": 0.7640736}
  ]
}
```

### Health Check
```http
GET /ping
```