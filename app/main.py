from fastapi import FastAPI, BackgroundTasks, Query
from app.handlers import home_handler, embedding_handler
from app.tasks.ping import ping

app = FastAPI()

app.get("/")(home_handler.root)

app.get("/hello/{name}")(home_handler.say_hello)

@app.get("/ping")
def ping_task():
    task = ping.delay()
    return {"task_id": task.id, "status": "queued"}

# Embedding endpoint using handler pattern
app.post("/embeddings/generate")(embedding_handler.generate_embeddings)
