import os
from typing import List, Dict, Optional, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

class QdrantService:
    """
    Service for managing vector embeddings in Qdrant.
    Handles collection creation, vector storage, and similarity search.
    """
    
    def __init__(self):
        """
        Initialize the Qdrant client and ensure the collection exists.
        Qdrant connection parameters are taken from environment variables.
        """
        # Qdrant connection parameters
        qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        
        # Initialize client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        print(f"Qdrant bağlantısı kuruldu: {qdrant_host}:{qdrant_port}")
        
        # Ensure collection exists
        self.initialize_collection()
    
    def initialize_collection(self, collection_name: str = "fashion_products"):
        """
        Check if the collection exists and create it if it doesn't.
        
        Args:
            collection_name: Name of the collection to initialize
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if not exists:
                print(f"Creating collection '{collection_name}'")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=2048,  # ResNet50 embedding size
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Collection '{collection_name}' created successfully")
            else:
                print(f"Collection '{collection_name}' already exists")
                
        except Exception as e:
            print(f"Error initializing Qdrant collection: {str(e)}")
    
    def save_embedding(self, product_id: int, embedding: List[float]) -> bool:
        """
        Store a single product embedding in Qdrant.
        
        Args:
            product_id: Product ID
            embedding: Embedding vector (as list)
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            self.client.upsert(
                collection_name="fashion_products",
                points=[{
                    "id": product_id,
                    "vector": embedding,
                    "payload": {"product_id": product_id}
                }]
            )
            print(f"Ürün {product_id} için embedding Qdrant'a payload ile kaydedildi.")
            return True
        except Exception as e:
            print(f"Qdrant'a kaydetme hatası: {str(e)}")
            return False
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """
        Get a product embedding from Qdrant.
        
        Args:
            product_id: Product ID
            
        Returns:
            np.ndarray: Embedding vector or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name="fashion_products",
                ids=[product_id]
            )
            if result and result[0].vector is not None:
                return np.array(result[0].vector, dtype=np.float32)
            return None
        except Exception as e:
            print(f"Qdrant'tan embedding alma hatası: {str(e)}")
            return None
    
    def get_embeddings_batch(self, product_ids: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """
        Get multiple product embeddings from Qdrant.
        
        Args:
            product_ids: List of product IDs
            
        Returns:
            Dict: Dictionary mapping product IDs to their embedding vectors
        """
        if not product_ids:
            return {}
            
        result_dict = {}
        
        try:
            results = self.client.retrieve(
                collection_name="fashion_products",
                ids=product_ids,
                with_vectors=True
            )
            
            retrieved_vectors = {item.id: item.vector for item in results}
            
            for pid in product_ids:
                if pid in retrieved_vectors:
                    result_dict[pid] = np.array(retrieved_vectors[pid], dtype=np.float32)
                else:
                    result_dict[pid] = None
                    
            return result_dict
            
        except Exception as e:
            print(f"Qdrant'tan toplu alma hatası: {str(e)}")
            return {pid: None for pid in product_ids}
    
    def save_embeddings_batch(self, product_ids: List[int], embeddings: Union[np.ndarray, List[List[float]]]) -> bool:
        """
        Store multiple product embeddings in Qdrant.
        
        Args:
            product_ids: List of product IDs
            embeddings: Embedding vectors (numpy array or list of lists)
            
        Returns:
            bool: Whether the operation was successful
        """
        if len(product_ids) == 0 or len(product_ids) != len(embeddings):
            return False
            
        try:
            points = []
            for i, product_id_val in enumerate(product_ids):
                embedding_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
                
                # Basic check for embedding format, can be enhanced
                if not isinstance(embedding_list, list) or not embedding_list:
                    print(f"[WARNING] Invalid or empty embedding for product_id {product_id_val}. Skipping.")
                    continue
                if not all(isinstance(x, (float, int)) for x in embedding_list):
                    print(f"[WARNING] Embedding for product_id {product_id_val} contains non-float/int values. Skipping.")
                    continue

                points.append({
                    "id": product_id_val,
                    "vector": embedding_list,
                    "payload": {"product_id": product_id_val}  # Added payload
                })
            
            if not points: 
                print("[INFO] No valid points to save in batch after filtering.")
                return False # Or True, depending on whether this is considered a success

            self.client.upsert(
                collection_name="fashion_products",
                points=points
            )
            
            print(f"{len(points)} ürün için embeddingler Qdrant'a payload ile kaydedildi.")
            return True
        except Exception as e:
            print(f"Qdrant'a toplu kaydetme hatası: {str(e)}")
            return False
    
    def delete_embedding(self, product_id: int) -> bool:
        """
        Delete a product embedding from Qdrant.
        
        Args:
            product_id: Product ID
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            self.client.delete(
                collection_name="fashion_products",
                points_selector=models.PointIdsList(
                    points=[product_id]
                )
            )
            print(f"Ürün {product_id} için embedding Qdrant'tan silindi.")
            return True
        except Exception as e:
            print(f"Qdrant'tan silme hatası: {str(e)}")
            return False
    
    def find_similar_products(self, product_id: int, top_n: int = 5) -> List[Dict]:
        """
        Find similar products using Qdrant vector search.
        
        Args:
            product_id: Product ID to find similar items for
            top_n: Number of similar products to return
            
        Returns:
            List of similar product IDs and similarity scores
        """
        try:
            search_result = self.client.search(
                collection_name="fashion_products",
                query_vector_id=product_id,
                limit=top_n + 1
            )
            print(f"Qdrant benzer ürün arama sonucu: {search_result}")
            similar_products = []

            for result in search_result:
                result_id = result.id if isinstance(result.id, int) else int(result.id)
                if result_id != product_id:
                    similar_products.append({
                        'id': result_id,
                        'similarity': result.score
                    })
            
            return similar_products[:top_n]
        except Exception as e:
            print(f"Qdrant benzer ürün arama hatası: {str(e)}")
            return []