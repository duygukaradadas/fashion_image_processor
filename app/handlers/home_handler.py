import os


async def root() -> dict:
    app_name: str = os.getenv("APP_NAME")

    return {
        "message": "Hello World",
        "appName": app_name,
    }

async def say_hello(name: str):
    return {"message": f"Hello {name}"}
