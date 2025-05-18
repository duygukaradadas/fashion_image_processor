import os
from typing import Dict, List, Optional, Any
import httpx
from pydantic import BaseModel
from urllib.parse import urljoin
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import logging
from dataclasses import dataclass
from functools import lru_cache
import time


class ProductResponse(BaseModel):
    """Response model for product data"""
    id: int
    image_url: str


class PaginationLink(BaseModel):
    """Model for pagination link data"""
    url: Optional[str] = None
    label: str
    active: bool


class PaginationMeta(BaseModel):
    """Model for pagination metadata"""
    current_page: int
    from_: int = None
    last_page: int
    links: List[PaginationLink]
    path: str
    per_page: int
    to: int
    total: int

    model_config = {
        'populate_by_name': True,
        'alias_generator': lambda x: 'from' if x == 'from_' else x
    }


class PaginationLinks(BaseModel):
    """Model for pagination links"""
    first: str
    last: str
    prev: Optional[str] = None
    next: Optional[str] = None


class ProductsListResponse(BaseModel):
    """Complete response model for products list"""
    data: List[ProductResponse]
    links: PaginationLinks
    meta: PaginationMeta


@dataclass
class Product:
    id: int
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None


@dataclass
class PaginatedResponse:
    data: List[Product]
    meta: Dict[str, Any]
    links: Dict[str, Any]


class ApiClient:
    """Client for connecting to the product API"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API. Defaults to environment variable API_BASE_URL 
        """
        self.base_url = base_url or "https://fashion.aknevrnky.dev/api"
        self.session = None
        self.logger = logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(50)  # Increased from 100 to 50 for better control
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        
    async def _ensure_session(self):
        """Ensure that a session exists and is valid."""
        if self.session is None or self.session.closed:
            # Configure connection pooling with optimized settings
            connector = aiohttp.TCPConnector(
                limit=50,  # Reduced from 100 to 50 for better control
                ttl_dns_cache=300,
                use_dns_cache=True,
                force_close=False,
                enable_cleanup_closed=True,
                ssl=False  # Disable SSL verification for better performance
            )
            timeout = aiohttp.ClientTimeout(total=10)  # Reduced from 30 to 10 seconds
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "accept": "application/json",
                    "Authorization": "Bearer fashion-api-token-2025"
                }
            )
        
    async def __aenter__(self):
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make an HTTP request with retry logic and connection pooling."""
        await self._ensure_session()
            
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Check cache
        cache_key = f"{method}:{url}"
        if cache_key in self._cache:
            cache_time, cache_data = self._cache[cache_key]
            if time.time() - cache_time < self._cache_ttl:
                return cache_data
        
        async with self._semaphore:  # Limit concurrent connections
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Cache successful responses
                    self._cache[cache_key] = (time.time(), data)
                    return data
            except aiohttp.ClientError as e:
                self.logger.error(f"API request failed: {str(e)}")
                raise
            
    async def get_products(self, page: int = 1) -> PaginatedResponse:
        """Get products with pagination and caching."""
        data = await self._make_request("GET", f"products?page={page}")
        return PaginatedResponse(
            data=[Product(**p) for p in data["data"]],
            meta=data["meta"],
            links=data["links"]
        )
        
    async def get_product_image(self, product_id: int) -> Optional[bytes]:
        """Get product image with retry logic and connection pooling."""
        try:
            # First get the product data to get the image_url
            product_data = await self._make_request("GET", f"products/{product_id}")
            if not product_data or "data" not in product_data or "image_url" not in product_data["data"]:
                self.logger.error(f"No image URL found for product {product_id}")
                return None
                
            image_url = product_data["data"]["image_url"]
            
            # Now fetch the image from the image_url
            await self._ensure_session()
            async with self._semaphore:  # Limit concurrent connections
                async with self.session.get(image_url) as response:
                    if response.status == 200:
                        return await response.read()
                    self.logger.error(f"Failed to fetch image from {image_url}, status: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Failed to get image for product {product_id}: {str(e)}")
            return None
            
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
