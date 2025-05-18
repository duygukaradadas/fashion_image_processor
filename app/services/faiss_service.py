import numpy as np
import faiss
import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

class FAISSService:
    """
    Service for managing vector embeddings using FAISS.
    Handles vector storage, similarity search, and persistence.
    """
    
    def __init__(self, dimension: int = 2048):
        """
        Initialize the FAISS index and storage.
        
        Args:
            dimension: Size of the embedding vectors (default: 2048 for ResNet50)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.product_ids = []  # Maps FAISS index to product IDs
        self.index_path = Path("data/faiss")
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing index if available
        self.load_index()
        print(f"FAISS index initialized with {len(self.product_ids)} products")
    
    def save_embedding(self, product_id: int, embedding: List[float]) -> bool:
        """
        Store a single product embedding in FAISS.
        
        Args:
            product_id: Product ID
            embedding: Embedding vector (as list)
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            # Validate embedding
            if not isinstance(embedding, (list, np.ndarray)) or len(embedding) != self.dimension:
                raise ValueError(f"Invalid embedding format or size. Expected {self.dimension} dimensions")
            
            # Convert to numpy array and reshape for FAISS
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            # Add to index
            self.index.add(embedding_array)
            self.product_ids.append(product_id)
            
            # Save index after each addition
            self.save_index()
            
            print(f"Embedding saved for product {product_id}")
            return True
            
        except Exception as e:
            print(f"Error saving embedding: {str(e)}")
            return False
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """
        Get a product embedding from FAISS.
        
        Args:
            product_id: Product ID
            
        Returns:
            np.ndarray: Embedding vector or None if not found
        """
        try:
            if product_id not in self.product_ids:
                return None
                
            idx = self.product_ids.index(product_id)
            return self.index.reconstruct(idx)
            
        except Exception as e:
            print(f"Error retrieving embedding: {str(e)}")
            return None
    
    def find_similar_products(self, embedding: Union[List[float], np.ndarray], top_n: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar products using vector similarity search.
        
        Args:
            embedding: Query embedding vector
            top_n: Number of similar products to return
            score_threshold: Minimum similarity score threshold (0-1)
            
        Returns:
            List of dictionaries containing product IDs and similarity scores
        """
        try:
            # Convert embedding to numpy array
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            embedding = embedding.reshape(1, -1)
            
            # Search for similar vectors
            distances, indices = self.index.search(embedding, top_n)
            
            # Format results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # FAISS returns -1 for invalid indices
                    # Convert L2 distance to similarity score (1 / (1 + distance))
                    score = 1.0 / (1.0 + dist)
                    if score >= score_threshold:
                        results.append({
                            "product_id": self.product_ids[idx],
                            "score": float(score)  # Convert to float for JSON serialization
                        })
            
            return results
            
        except Exception as e:
            print(f"Error finding similar products: {str(e)}")
            return []
    
    def find_similar_by_id(self, product_id: int, top_n: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar products using a product ID as reference.
        
        Args:
            product_id: Reference product ID
            top_n: Number of similar products to return
            score_threshold: Minimum similarity score threshold (0-1)
            
        Returns:
            List of dictionaries containing product IDs and similarity scores
        """
        try:
            # Get the reference product's embedding
            embedding = self.get_embedding(product_id)
            if embedding is None:
                print(f"No embedding found for product {product_id}")
                return []
            
            # Find similar products
            return self.find_similar_products(embedding, top_n, score_threshold)
            
        except Exception as e:
            print(f"Error finding similar products by ID: {str(e)}")
            return []
    
    def save_embeddings_batch(self, product_ids: List[int], embeddings: Union[np.ndarray, List[List[float]]]) -> bool:
        """
        Store multiple product embeddings in FAISS.
        
        Args:
            product_ids: List of product IDs
            embeddings: Embedding vectors (numpy array or list of lists)
            
        Returns:
            bool: Whether the operation was successful
        """
        if len(product_ids) == 0 or len(product_ids) != len(embeddings):
            return False
            
        try:
            # Convert to numpy array if needed
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            # Add to index
            self.index.add(embeddings)
            self.product_ids.extend(product_ids)
            
            # Save index after batch addition
            self.save_index()
            
            print(f"Saved {len(product_ids)} embeddings to FAISS")
            return True
            
        except Exception as e:
            print(f"Error saving batch embeddings: {str(e)}")
            return False
    
    def delete_embedding(self, product_id: int) -> bool:
        """
        Delete a product embedding from FAISS.
        Note: FAISS doesn't support direct deletion, so we rebuild the index.
        
        Args:
            product_id: Product ID
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            if product_id not in self.product_ids:
                return False
                
            # Get all embeddings except the one to delete
            keep_indices = [i for i, pid in enumerate(self.product_ids) if pid != product_id]
            if not keep_indices:
                # If deleting the last embedding, reset the index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.product_ids = []
            else:
                # Rebuild index with remaining embeddings
                new_index = faiss.IndexFlatL2(self.dimension)
                embeddings = [self.index.reconstruct(i) for i in keep_indices]
                new_index.add(np.array(embeddings))
                self.index = new_index
                self.product_ids = [self.product_ids[i] for i in keep_indices]
            
            # Save the updated index
            self.save_index()
            
            print(f"Deleted embedding for product {product_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting embedding: {str(e)}")
            return False
    
    def get_collection_count(self) -> int:
        """
        Get the total number of points in the index.
        
        Returns:
            int: Number of points in the index
        """
        return len(self.product_ids)
    
    def get_products_with_embeddings(self, limit: int = 5) -> List[int]:
        """
        Get a list of product IDs that have embeddings.
        
        Args:
            limit: Maximum number of IDs to return
            
        Returns:
            List of product IDs
        """
        return self.product_ids[:limit]
    
    def save_index(self):
        """Save the FAISS index and product IDs to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
            
            # Save product IDs
            with open(self.index_path / "product_ids.json", "w") as f:
                json.dump(self.product_ids, f)
                
            print(f"Index saved with {len(self.product_ids)} products")
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            # Don't raise the exception, just log it
            # This allows the process to continue even if saving fails
    
    def load_index(self):
        """Load the FAISS index and product IDs from disk."""
        try:
            index_file = self.index_path / "index.faiss"
            ids_file = self.index_path / "product_ids.json"
            
            if index_file.exists() and ids_file.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load product IDs
                with open(ids_file, "r") as f:
                    self.product_ids = json.load(f)
                    
                print(f"Loaded index with {len(self.product_ids)} products")
                
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            # Initialize new index if loading fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.product_ids = [] 