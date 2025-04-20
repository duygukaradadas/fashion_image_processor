import os
from typing import Dict, List, Optional, Any
import httpx
from pydantic import BaseModel


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

    class Config:
        fields = {
            'from_': 'from'
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


class ApiClient:
    """Client for connecting to the product API"""
    
    def __init__(self, base_url: str = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API. Defaults to environment variable API_BASE_URL 
        """
        self.base_url = base_url or os.getenv('API_BASE_URL')
        
    async def get_products(self, page: int = 1) -> ProductsListResponse:
        """
        Fetch products from the API.
        
        Args:
            page: Page number to fetch. Defaults to 1.
            
        Returns:
            A ProductsListResponse object containing product data and pagination info.
            
        Raises:
            httpx.HTTPError: If an HTTP error occurs
            ValueError: If the response cannot be parsed
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/products",
                params={"page": page}
            )
            response.raise_for_status()
            
            data = response.json()
            return ProductsListResponse(**data)
            
    async def get_product(self, product_id: int) -> ProductResponse:
        """
        Fetch a single product by ID.
        
        Args:
            product_id: The ID of the product to fetch
            
        Returns:
            A ProductResponse object containing product data
            
        Raises:
            httpx.HTTPError: If an HTTP error occurs
            ValueError: If the response cannot be parsed
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/products/{product_id}"
            )
            response.raise_for_status()
            
            data = response.json()
            if "data" in data:
                return ProductResponse(**data["data"])
            return ProductResponse(**data)
            
    async def get_product_image(self, product_id: int = None, image_url: str = None) -> bytes:
        """
        Fetch the image data for a product.
        This method is suitable for training services that need the raw image data.
        
        Args:
            product_id: The ID of the product to fetch the image for. 
                        If provided, the product details will be fetched first to get the image URL.
            image_url: Direct URL to the image. If provided, product_id is ignored.
            
        Returns:
            Bytes containing the image data that can be used directly with image processing 
            libraries like PIL/Pillow, OpenCV, or TensorFlow.
            
        Raises:
            ValueError: If neither product_id nor image_url is provided
            httpx.HTTPError: If an HTTP error occurs
        """
        if not product_id and not image_url:
            raise ValueError("Either product_id or image_url must be provided")
            
        if not image_url and product_id:
            # Get the product details to retrieve the image URL
            product = await self.get_product(product_id)
            image_url = product.image_url
            
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            
            return response.content
