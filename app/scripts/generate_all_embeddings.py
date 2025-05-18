import requests
import json
import time
from typing import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_all_product_ids(limit: int = 20000) -> List[int]:
    """Fetch all product IDs up to the specified limit."""
    all_ids = []
    page = 1
    per_page = 100  # API's default page size
    session = create_session()
    
    while len(all_ids) < limit:
        url = f"https://fashion.aknevrnky.dev/api/products?page={page}"
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer fashion-api-token-2025"
        }
        
        try:
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            products = data.get("data", [])
            if not products:
                print("No more products found.")
                break
                
            new_ids = [product["id"] for product in products]
            all_ids.extend(new_ids)
            print(f"Fetched {len(all_ids)} products so far...")
            
            if len(all_ids) >= limit:
                all_ids = all_ids[:limit]
                break
                
            page += 1
            time.sleep(0.5)  # Increased delay to avoid rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {str(e)}")
            time.sleep(5)  # Wait longer on error
            continue
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing response for page {page}: {str(e)}")
            print(f"Response content: {response.text[:200]}...")  # Print first 200 chars of response
            time.sleep(5)
            continue
    
    return all_ids

def generate_embeddings_batch(product_ids: List[int], batch_size: int = 50):
    """Generate embeddings for a batch of products."""
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer fashion-api-token-2025",
        "Content-Type": "application/json"
    }
    session = create_session()
    
    for i in range(0, len(product_ids), batch_size):
        batch = product_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}, products {i+1} to {min(i+batch_size, len(product_ids))}")
        
        try:
            response = session.post(
                "http://localhost:4000/embeddings/generate",
                headers=headers,
                json=batch,  # Send the list directly
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "task_id" in result:
                print(f"Successfully queued batch {i//batch_size + 1} (task_id: {result['task_id']})")
            else:
                print(f"Unexpected response format for batch {i//batch_size + 1}: {result}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            time.sleep(5)  # Wait longer on error
            continue
        except json.JSONDecodeError as e:
            print(f"Error parsing response for batch {i//batch_size + 1}: {str(e)}")
            print(f"Response content: {response.text[:200]}...")  # Print first 200 chars of response
            time.sleep(5)
            continue
        
        # Wait between batches to avoid overwhelming the system
        time.sleep(2)

def main():
    print("Fetching all product IDs...")
    all_ids = get_all_product_ids()
    print(f"Found {len(all_ids)} products")
    
    if not all_ids:
        print("No product IDs found. Exiting.")
        return
    
    print("Generating embeddings in batches...")
    generate_embeddings_batch(all_ids)
    print("Finished queuing all embedding generation tasks")

if __name__ == "__main__":
    main() 