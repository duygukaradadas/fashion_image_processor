Run the following command to start the Celery worker:

```shell
celery -A app.celery_config.celery_app worker --loglevel=info
```
