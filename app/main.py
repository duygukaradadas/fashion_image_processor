from fastapi import FastAPI
from app.handlers import home_handler

app = FastAPI()

app.get("/")(home_handler.root)

app.get("/hello/{name}")(home_handler.say_hello)
