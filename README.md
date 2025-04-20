Run the following command to start the Celery worker:

```shell
celery -A app.celery_config.celery_app worker --loglevel=info
```


# Create the client
client = ApiClient(base_url="https://fashion.aknevrnky.dev")

# Get image bytes by product ID
image_data = await client.get_product_image(product_id=1)

# Or directly from a URL
image_data = await client.get_product_image(image_url="https://fashion.aknevrnky.dev/storage/products/10000.jpg")

# Process with PIL/Pillow
from PIL import Image
import io
image = Image.open(io.BytesIO(image_data))
# Now the image can be processed for training