FROM python:3.13-alpine

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN python -m venv .venv

RUN source .venv/bin/activate

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0"]
