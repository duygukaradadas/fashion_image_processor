from fastapi import FastAPI, BackgroundTasks, Query
from app.handlers import home_handler, embedding_handler
from app.tasks.ping import ping
from app.services.redis_service import RedisService

app = FastAPI()

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
    redis_service = RedisService()
    similar_products = redis_service.find_similar_products(product_id=product_id, top_n=top_n)
    return {"similar_products": similar_products}
