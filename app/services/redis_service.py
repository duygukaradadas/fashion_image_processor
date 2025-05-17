# app/services/redis_service.py
import os
import json
import numpy as np
from typing import List, Dict, Optional, Any
import redis

# Default DSN if not provided by environment variable
DEFAULT_APP_REDIS_DSN = 'redis://redis:6379/0'
APP_REDIS_DSN_ENV_VAR = 'APP_REDIS_DSN'

class RedisService:
    """
    Redis'te embedding vektörlerini saklamak ve sorgulamak için servis.
    """
    
    def __init__(self):
        """
        Redis bağlantısını başlat.
        Redis DSN'ini çevre değişkeninden al.
        """
        app_redis_dsn = os.getenv(APP_REDIS_DSN_ENV_VAR, DEFAULT_APP_REDIS_DSN)
        
        try:
            # redis-py can directly use a DSN string
            self.redis = redis.from_url(app_redis_dsn, decode_responses=True)
            # Ensure connection is working
            self.redis.ping()
            print(f"Redis bağlantısı {app_redis_dsn} üzerinden kuruldu.")
        except redis.exceptions.ConnectionError as e:
            print(f"Redis DSN ({app_redis_dsn}) ile bağlantı hatası: {e}. Lütfen APP_REDIS_DSN ortam değişkenini kontrol edin.")
            # Depending on desired behavior, you might want to raise the error
            # or have a fallback or a non-functional Redis client.
            # For now, it will print an error and subsequent operations might fail.
            self.redis = None # Or some mock/dummy client if preferred for graceful degradation
        except Exception as e: # Catch other potential errors from from_url, e.g., bad DSN format
            print(f"Redis DSN ({app_redis_dsn}) parse edilirken veya bağlantı kurulurken hata: {e}")
            self.redis = None

    def save_embedding(self, product_id: int, embedding: List[float]) -> bool:
        """
        Ürün embedding'ini Redis'e kaydet.
        
        Args:
            product_id: Ürün ID'si
            embedding: Embedding vektörü (liste olarak)
            
        Returns:
            bool: Başarılı olup olmadığı
        """
        if not self.redis:
            print("Redis bağlantısı yok, embedding kaydedilemedi.")
            return False
        try:
            key = f"embedding:{product_id}"
            
            # Embedding'i JSON olarak serialize ederek saklayalım
            embedding_json = json.dumps(embedding)
            
            # Redis'e kaydet
            result = self.redis.set(key, embedding_json)
            
            if result:
                print(f"Ürün {product_id} için embedding Redis'e kaydedildi.")
            else:
                print(f"Ürün {product_id} için embedding kaydedilemedi!")
                
            return result
        except Exception as e:
            print(f"Redis'e kaydetme hatası: {str(e)}")
            return False
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """
        Ürün embedding'ini Redis'ten al.
        
        Args:
            product_id: Ürün ID'si
            
        Returns:
            np.ndarray: Embedding vektörü veya None
        """
        if not self.redis:
            print("Redis bağlantısı yok, embedding alınamadı.")
            return None
        try:
            key = f"embedding:{product_id}"
            
            # Redis'ten veriyi al
            embedding_json = self.redis.get(key)
            
            if not embedding_json:
                print(f"Ürün {product_id} için embedding Redis'te bulunamadı.")
                return None
                
            # JSON'dan listeye dönüştür
            embedding_list = json.loads(embedding_json)
            
            # Numpy array'e dönüştür
            embedding = np.array(embedding_list, dtype=np.float32)
            
            return embedding
        except Exception as e:
            print(f"Redis'ten alma hatası: {str(e)}")
            return None
    
    def delete_embedding(self, product_id: int) -> bool:
        """
        Ürün embedding'ini Redis'ten sil.
        
        Args:
            product_id: Ürün ID'si
            
        Returns:
            bool: Silme işleminin başarılı olup olmadığı
        """
        if not self.redis:
            print("Redis bağlantısı yok, embedding silinemedi.")
            return False
        try:
            key = f"embedding:{product_id}"
            
            # Redis'ten sil, sonuç olarak silinen key sayısını döndürür (0 veya 1)
            result = self.redis.delete(key)
            
            if result > 0:
                print(f"Ürün {product_id} için embedding Redis'ten silindi.")
                return True
            else:
                print(f"Ürün {product_id} için embedding Redis'te bulunamadı veya silinemedi.")
                return False
                
        except Exception as e:
            print(f"Redis'ten silme hatası: {str(e)}")
            return False
    
    def find_similar_products(self, product_id: int, top_n: int = 5) -> List[Dict]:
        """
        Belirli bir ürün ID'si için benzer ürünleri bul.
        Bu basit implementasyon tüm embeddingleri alıp bellek üzerinde kosinüs benzerliği hesaplar.
        
        Args:
            product_id: Benzer ürünleri bulmak için ürün ID'si
            top_n: Döndürülecek benzer ürün sayısı
            
        Returns:
            Benzer ürün ID'leri ve benzerlik skorlarını içeren sözlük listesi
        """
        if not self.redis:
            print("Redis bağlantısı yok, benzer ürünler bulunamadı.")
            return []
        try:
            # Sorgu ürünü için embedding al
            query_embedding = self.get_embedding(product_id)
            
            if query_embedding is None:
                raise ValueError(f"Ürün ID {product_id} Redis'te bulunamadı")
            
            # Tüm ürünlerin ID'lerini al (pattern matching ile)
            all_keys = self.redis.keys("embedding:*")
            product_ids = [int(key.split(':')[1]) for key in all_keys]
            
            # Benzerlik skorlarını hesapla
            similarity_scores = []
            
            for pid in product_ids:
                # Kendisini atla
                if pid == product_id:
                    continue
                    
                # Embedding'i al
                product_embedding = self.get_embedding(pid)
                
                if product_embedding is None:
                    continue
                    
                # Kosinüs benzerliğini hesapla
                similarity = self._cosine_similarity(query_embedding, product_embedding)
                
                similarity_scores.append({
                    'id': pid,
                    'similarity': float(similarity)
                })
                
            # Benzerlik skoruna göre sırala (en yüksek önce)
            similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
            
            # En yüksek N sonucu döndür
            return similarity_scores[:top_n]
        except Exception as e:
            print(f"Benzer ürünler bulma hatası: {str(e)}")
            return []
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        İki embedding arasındaki kosinüs benzerliğini hesapla.
        
        Args:
            embedding1: Birinci embedding vektörü
            embedding2: İkinci embedding vektörü
            
        Returns:
            float: Kosinüs benzerlik skoru (0-1)
        """
        if not self.redis:
            print("Redis bağlantısı yok, kosinüs benzerliği hesaplanamadı.")
            return 0
        try:
            # NumPy dizilerine dönüştür (eğer değilse)
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            # Kosinüs benzerliğini hesapla
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Kosinüs benzerliği hesaplama hatası: {str(e)}")
            return 0