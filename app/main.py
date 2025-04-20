from fastapi import FastAPI
from app.handlers import home_handler
from app.tasks.ping import ping

app = FastAPI()

app.get("/")(home_handler.root)

app.get("/hello/{name}")(home_handler.say_hello)

@app.get("/ping")
def ping_task():
    task = ping.delay()
    return {"task_id": task.id, "status": "queued"}
