import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

API_AUTH_TOKEN_ENV = "API_AUTH_TOKEN"

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.api_key = os.getenv(API_AUTH_TOKEN_ENV)
        if not self.api_key:
            print(f"WARNING: {API_AUTH_TOKEN_ENV} environment variable not set. API authentication will fail.")
            # Or raise an error if the app should not start without it
            # raise ValueError(f"{API_AUTH_TOKEN_ENV} environment variable is required.")


    async def dispatch(self, request: Request, call_next):
        # Allow health check endpoint without authentication
        if request.url.path in ["/embeddings/health", "/health"]:
            return await call_next(request)

        if not self.api_key:
            # If API_KEY is not set, all authenticated requests will fail.
            # This ensures the application doesn\'t run in an insecure default state.
            return JSONResponse(
                status_code=500, 
                content={"detail": "API authentication is not configured on the server."}
            )

        authorization: str = request.headers.get("Authorization")
        error_response = JSONResponse(
            status_code=401,
            content={"detail": "Not authenticated"},
            headers={"WWW-Authenticate": "Bearer"},
        )

        if not authorization:
            # Allowing certain paths to be exempt (e.g., health checks, docs)
            # For simplicity, now all paths require auth. Add exemptions if needed.
            # if request.url.path in ["/docs", "/openapi.json", "/ping", "/"]: # Example exemptions
            #     return await call_next(request)
            return error_response

        try:
            scheme, token = authorization.split()
            if scheme.lower() != 'bearer':
                return error_response
        except ValueError:
            # Authorization header is malformed
            return error_response
        
        if token != self.api_key:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        
        response = await call_next(request)
        return response 