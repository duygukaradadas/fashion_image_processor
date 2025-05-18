from fastapi import FastAPI, Query, UploadFile, File
from app.handlers import home_handler
from app.services.faiss_service import FAISSService
from app.middleware import APIKeyAuthMiddleware
from app.handlers import embedding_handler as embedding_routes
from fastapi.middleware.cors import CORSMiddleware
from app.services.embedding_service import EmbeddingService
from PIL import Image
import io
  
app = FastAPI(debug=True)

# Add the authentication middleware
# app.add_middleware(APIKeyAuthMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:8090"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = EmbeddingService()

# Include embedding routes
app.include_router(embedding_routes.router, prefix="/embeddings", tags=["embeddings"])
# Basic endpoints
app.get("/")(home_handler.root)
app.get("/hello/{name}")(home_handler.say_hello)

@app.post("/similar-products/upload")
async def find_similar_products_from_image(
    image: UploadFile = File(...),
    top_n: int = Query(default=5, description="Number of similar products to return"),
    score_threshold: float = Query(default=0.001, description="Minimum similarity score threshold (0-1)")
):
    """Find similar products based on an uploaded image."""
    try:
        # Read and validate the uploaded image
        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        except Exception as e:
            return {"error": f"Invalid image format: {str(e)}"}
        
        # Generate embedding for the uploaded image
        embedding = embedding_service.model.get_embedding(pil_image)
        if embedding is None:
            return {"error": "Failed to generate embedding for the image"}
        
        # Find similar products
        similar_products = embedding_service.find_similar_products(
            embedding,
            top_n=top_n * 2,  # Request more results to account for duplicates
            score_threshold=score_threshold
        )
        
        # Remove duplicates while preserving order and keeping highest score
        seen_ids = set()
        unique_products = []
        for product in similar_products:
            product_id = product["product_id"]
            if product_id not in seen_ids:
                seen_ids.add(product_id)
                unique_products.append(product)
                if len(unique_products) >= top_n:
                    break
        
        # Transform the response to include image URLs
        response = []
        for product in unique_products:
            response.append({
                "id": product["product_id"],
                "similarity_score": product["score"],
                "image_url": f"https://fashion.aknevrnky.dev/api/products/{product['product_id']}/image"
            })
        
        return {"similar_products": response}
        
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

@app.get("/similar-products/{product_id}")
async def get_similar_products(
    product_id: int,
    top_n: int = Query(default=5, description="Number of similar products to return"),
    score_threshold: float = Query(default=0.1, description="Minimum similarity score threshold (0-1)")
):
    """Find similar products based on image embeddings."""
    faiss_service = FAISSService()
    similar_products = faiss_service.find_similar_by_id(product_id, top_n * 2, score_threshold)  # Request more results to account for duplicates
    
    # Remove duplicates while preserving order and keeping highest score
    seen_ids = set()
    unique_products = []
    for product in similar_products:
        product_id = product["product_id"]
        if product_id not in seen_ids:
            seen_ids.add(product_id)
            unique_products.append(product)
            if len(unique_products) >= top_n:
                break
    
    # Transform the response to include image URLs
    response = []
    for product in unique_products:
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