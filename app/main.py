from fastapi import FastAPI, Query, UploadFile, File
from app.handlers import home_handler, embedding_handler
from app.tasks.ping import ping
from app.services.redis_service import RedisService
from app.services.qdrant_service import QdrantService
from app.middleware import APIKeyAuthMiddleware
from app.resnet.model import ResNetEmbedder
import numpy as np

app = FastAPI()

# Add the authentication middleware
app.add_middleware(APIKeyAuthMiddleware)

app.get("/")(home_handler.root)

app.get("/hello/{name}")(home_handler.say_hello)

@app.get("/ping")
def ping_task():
    task = ping.delay()
    return {"task_id": task.id, "status": "queued"}

# Embedding endpoints
app.post("/embeddings/generate")(embedding_handler.generate_embeddings)
app.post("/embeddings/product/{product_id}")(embedding_handler.generate_single_embedding)
app.put("/embeddings/product/{product_id}")(embedding_handler.update_single_embedding)
app.delete("/embeddings/product/{product_id}")(embedding_handler.delete_single_embedding)

@app.get("/similar-products/{product_id}")
async def get_similar_products(
    product_id: int,
    top_n: int = Query(default=5, description="Number of similar products to return")
):
    """Find similar products based on image embeddings."""
    qdrant_service = QdrantService()
    similar_products = qdrant_service.find_similar_products(product_id=product_id, top_n=top_n)
    return {"similar_products": similar_products}

@app.post("/similar-products/from-image")
async def find_similar_products_from_image(
    image: UploadFile = File(...),
    top_n: int = Query(default=5, description="Number of similar products to return")
):
    """
    Find similar products based on an uploaded image.
    The image will be transformed into a 2048-sized vector using ResNet50.
    """
    # Read the uploaded image
    image_data = await image.read()
    
    # Convert image to embedding using ResNet50
    embedder = ResNetEmbedder()
    embedding = embedder.get_embedding(image_data)
    
    # Pass the embedding list directly to find_similar_products
    qdrant_service = QdrantService()
    similar_products = qdrant_service.find_similar_products(
        product_id=embedding.tolist(),  # Pass embedding instead of product_id
        top_n=top_n
    )
    
    return {"similar_products": similar_products}

@app.post("/admin/migrate-to-qdrant")
async def migrate_to_qdrant():
    """Migrate all existing embeddings from Redis to Qdrant."""
    redis_service = RedisService()
    qdrant_service = QdrantService()
    
    # Get all product IDs from Redis
    all_keys = redis_service.redis.keys("embedding:*")
    product_ids = [int(key.split(':')[1]) for key in all_keys]
    
    batch_size = 100
    total_migrated = 0
    
    # Process in batches
    for i in range(0, len(product_ids), batch_size):
        batch_ids = product_ids[i:i+batch_size]
        
        # Get embeddings from Redis
        embeddings_dict = {}
        for pid in batch_ids:
            embedding = redis_service.get_embedding(pid)
            if embedding is not None:
                embeddings_dict[pid] = embedding
        
        # Convert to format needed for Qdrant batch save
        batch_product_ids = list(embeddings_dict.keys())
        batch_embeddings = [embeddings_dict[pid] for pid in batch_product_ids]
        
        # FIX: Convert to numpy array before passing to Qdrant
        if batch_product_ids:
            batch_embeddings_np = np.array(batch_embeddings)
            try:
                success = qdrant_service.save_embeddings_batch(
                    product_ids=batch_product_ids,
                    embeddings=batch_embeddings_np
                )
                if success:
                    total_migrated += len(batch_product_ids)
            except Exception as e:
                print(f"Error saving batch to Qdrant: {str(e)}")
    
    return {"status": "completed", "total_migrated": total_migrated}