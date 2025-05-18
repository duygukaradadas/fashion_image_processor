from fastapi import FastAPI, Query
from app.handlers import home_handler
from app.services.faiss_service import FAISSService
from app.middleware import APIKeyAuthMiddleware
from app.handlers import embedding_handler as embedding_routes
  
app = FastAPI(debug=True)

# Add the authentication middleware
app.add_middleware(APIKeyAuthMiddleware)

# Include embedding routes
app.include_router(embedding_routes.router, prefix="/embeddings", tags=["embeddings"])
# Basic endpoints
app.get("/")(home_handler.root)
app.get("/hello/{name}")(home_handler.say_hello)

@app.get("/similar-products/{product_id}")
async def get_similar_products(
    product_id: int,
    top_n: int = Query(default=5, description="Number of similar products to return"),
    score_threshold: float = Query(default=0.1, description="Minimum similarity score threshold (0-1)")
):
    """Find similar products based on image embeddings."""
    faiss_service = FAISSService()
    similar_products = faiss_service.find_similar_by_id(product_id, top_n, score_threshold)
    
    # Transform the response to include image URLs
    response = []
    for product in similar_products:
        response.append({
            "id": product["product_id"],
            "similarity_score": product["score"],
            "image_url": f"https://fashion.aknevrnky.dev/api/products/{product['product_id']}/image"
        })
    
    return {"similar_products": response}

@app.get("/embeddings/count")
async def get_embedding_count():
    """Get the total number of embeddings in FAISS."""
    faiss_service = FAISSService()
    count = faiss_service.get_collection_count()
    return {"total_embeddings": count}

@app.get("/embeddings/check/{product_id}")
async def check_embedding(product_id: int):
    """Check if a product has an embedding in FAISS."""
    faiss_service = FAISSService()
    embedding = faiss_service.get_embedding(product_id)
    return {"has_embedding": embedding is not None}

@app.get("/embeddings/products")
async def get_products_with_embeddings(limit: int = Query(default=5, description="Maximum number of products to return")):
    """Get a list of product IDs that have embeddings."""
    faiss_service = FAISSService()
    product_ids = faiss_service.get_products_with_embeddings(limit)
    return {"product_ids": product_ids}