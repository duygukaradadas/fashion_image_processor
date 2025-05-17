import requests
import time
from typing import List

def get_all_product_ids() -> List[int]:
    """Get all product IDs from the API."""
    all_ids = []
    page = 1
    while True:
        response = requests.get(f"https://fashion.aknevrnky.dev/api/products?page={page}")
        data = response.json()
        
        # Add IDs from current page
        all_ids.extend([product['id'] for product in data['data']])
        
        # Check if there are more pages
        if not data['links']['next']:
            break
            
        page += 1
        print(f"Fetched page {page-1}, total IDs so far: {len(all_ids)}")
    
    return all_ids

def generate_embeddings_batch(product_ids: List[int], batch_size: int = 100):
    """Generate embeddings for a batch of products."""
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer fashion-api-token-2025",
        "Content-Type": "application/json"
    }
    
    for i in range(0, len(product_ids), batch_size):
        batch = product_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}, products {i+1} to {min(i+batch_size, len(product_ids))}")
        
        response = requests.post(
            "http://localhost:4000/embeddings/generate",
            headers=headers,
            json={"product_ids": batch}
        )
        
        if response.status_code == 200:
            print(f"Successfully queued batch {i//batch_size + 1}")
        else:
            print(f"Error processing batch {i//batch_size + 1}: {response.text}")
        
        # Wait a bit between batches to avoid overwhelming the system
        time.sleep(2)

def main():
    print("Fetching all product IDs...")
    all_ids = get_all_product_ids()
    print(f"Found {len(all_ids)} products")
    
    print("Generating embeddings in batches...")
    generate_embeddings_batch(all_ids)
    print("Finished queuing all embedding generation tasks")

if __name__ == "__main__":
    main() 