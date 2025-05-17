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
        Configure HNSW index for fast similarity search.
        
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
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0,  # Enable immediate indexing
                        memmap_threshold=20000,  # Use memory-mapped files for large collections
                        default_segment_number=2  # Optimize for parallel processing
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,  # Number of connections per element
                        ef_construct=100,  # Size of the dynamic candidate list
                        full_scan_threshold=10,  # Minimum allowed by Qdrant, always use HNSW index
                        max_indexing_threads=4,  # Number of threads for index construction
                        on_disk=False  # Keep index in memory for faster search
                    )
                )
                print(f"Collection '{collection_name}' created successfully with HNSW index")
            else:
                print(f"Collection '{collection_name}' already exists")
                # Update collection configuration to optimize for similarity search
                self.client.update_collection(
                    collection_name=collection_name,
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0,  # Enable immediate indexing
                        memmap_threshold=20000,  # Use memory-mapped files for large collections
                        default_segment_number=2  # Optimize for parallel processing
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,  # Number of connections per element
                        ef_construct=100,  # Size of the dynamic candidate list
                        full_scan_threshold=10,  # Minimum allowed by Qdrant, always use HNSW index
                        max_indexing_threads=4,  # Number of threads for index construction
                        on_disk=False  # Keep index in memory for faster search
                    )
                )
                print(f"Collection '{collection_name}' configuration updated for fast similarity search")
                
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
                    "id": int(product_id),  # Convert to integer
                    "vector": embedding
                }]
            )
            print(f"Ürün {product_id} için embedding Qdrant'a kaydedildi.")
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
                ids=[int(product_id)]
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
            # Convert product IDs to integers for Qdrant
            int_ids = [int(pid) for pid in product_ids]
            
            # Retrieve vectors from Qdrant
            results = self.client.retrieve(
                collection_name="fashion_products",
                ids=int_ids,
                with_vectors=True
            )
            
            # Create a mapping of id -> vector
            retrieved_vectors = {int(item.id): item.vector for item in results}
            
            # Map each requested ID to its vector or None
            for pid in product_ids:
                if pid in retrieved_vectors:
                    result_dict[pid] = np.array(retrieved_vectors[pid], dtype=np.float32)
                else:
                    result_dict[pid] = None
                    
            return result_dict
            
        except Exception as e:
            print(f"Qdrant'tan toplu alma hatası: {str(e)}")
            # Still provide a result for each ID, but with None values
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
            for i, product_id in enumerate(product_ids):
                # Convert to list if it's a numpy array
                embedding_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
                
                points.append({
                    "id": int(product_id),  # Convert to integer
                    "vector": embedding_list
                })
                
            self.client.upsert(
                collection_name="fashion_products",
                points=points
            )
            
            print(f"{len(points)} ürün için embeddingler Qdrant'a kaydedildi.")
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
                    points=[str(product_id)]
                )
            )
            print(f"Ürün {product_id} için embedding Qdrant'tan silindi.")
            return True
        except Exception as e:
            print(f"Qdrant'tan silme hatası: {str(e)}")
            return False
    
    def find_similar_products(self, product_id: int, top_n: int = 5) -> List[Dict]:
        """
        Find similar products using Qdrant vector search with optimized parameters.
        
        Args:
            product_id: Product ID to find similar items for
            top_n: Number of similar products to return
            
        Returns:
            List of similar product IDs and similarity scores
        """
        try:
            # Get the embedding for the query product
            embedding = self.get_embedding(product_id)
            if embedding is None:
                print(f"No embedding found for product {product_id}")
                return []
            
            # Get total number of points in collection
            collection_info = self.client.get_collection("fashion_products")
            total_points = collection_info.points_count
            
            # If we have fewer points than requested, adjust the limit
            search_limit = min(top_n + 1, total_points)
            
            # Search for similar products using the embedding with optimized parameters
            search_result = self.client.search(
                collection_name="fashion_products",
                query_vector=embedding.tolist(),
                limit=search_limit,
                search_params=models.SearchParams(
                    hnsw_ef=128,  # Size of the dynamic candidate list for HNSW search
                    exact=False  # Use approximate search for better performance
                ),
                score_threshold=0.7  # Only return results with similarity score above 0.7
            )
            
            # Filter out the query product and format results
            similar_products = []
            for result in search_result:
                result_id = int(result.id)
                if result_id != product_id:  # Skip the query product
                    similar_products.append({
                        'id': result_id,
                        'similarity': result.score,
                        'image_url': f"https://fashion.aknevrnky.dev/api/products/{result_id}/image"
                    })
            
            return similar_products[:top_n]
        except Exception as e:
            print(f"Qdrant benzer ürün arama hatası: {str(e)}")
            return []

    async def search_similar(self, embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        Search for similar vectors using a query vector.
        
        Args:
            embedding: Query vector to search with
            limit: Number of similar vectors to return
            
        Returns:
            List of similar vectors with their IDs and similarity scores
        """
        try:
            search_result = self.client.search(
                collection_name="fashion_products",
                query_vector=embedding,
                limit=limit
            )
            
            similar_products = []
            for result in search_result:
                similar_products.append({
                    'product_id': int(result.id),
                    'similarity_score': float(result.score),
                    'image_url': f"https://fashion.aknevrnky.dev/api/products/{result.id}/image"  # Construct image URL
                })
            
            return similar_products
        except Exception as e:
            print(f"Qdrant similarity search error: {str(e)}")
            return []