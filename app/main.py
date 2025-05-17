from fastapi import FastAPI, Query
from app.handlers import home_handler
from app.services.qdrant_service import QdrantService
from app.middleware import APIKeyAuthMiddleware
from app.routes import embedding_routes

app = FastAPI()

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
    top_n: int = Query(default=5, description="Number of similar products to return")
):
    """Find similar products based on image embeddings."""
    qdrant_service = QdrantService()
    similar_products = qdrant_service.find_similar_products(product_id, top_n)
    
    # Transform the response to include image URLs
    response = []
    for product in similar_products:
        response.append({
            "id": product["id"],
            "similarity_score": product["similarity"],
            "image_url": f"https://fashion.aknevrnky.dev/api/products/{product['id']}/image"
        })
    
    return {"similar_products": response}