import os

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    app_name: str = os.getenv("APP_NAME")

    return {
        "message": "Hello World",
        "appName": app_name,
    }


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
