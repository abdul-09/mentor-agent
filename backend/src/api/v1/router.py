"""
API V1 Router for AI Code Mentor
Implements RESTful resource naming and versioning (RULE API-001).

Compliance:
- RULE API-001: RESTful resource naming under /api/v1/
- RULE API-002: Correct HTTP status codes
- RULE API-003: Version management
"""

from fastapi import APIRouter

from src.api.v1.endpoints import auth, files, analysis, users, admin

# Create main API router for v1
api_v1_router = APIRouter()

# Include all endpoint routers with proper prefixes
api_v1_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_v1_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

api_v1_router.include_router(
    files.router,
    prefix="/files",
    tags=["file-management"]
)

api_v1_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["code-analysis"]
)

api_v1_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["administration"]
)


@api_v1_router.get("/", tags=["api-info"])
async def api_v1_info():
    """API v1 information endpoint."""
    return {
        "api_version": "1.0.0",
        "status": "active",
        "endpoints": [
            "/auth - Authentication and authorization",
            "/users - User management",
            "/files - File upload and management",
            "/analysis - Code and PDF analysis",
            "/admin - Administrative functions",
        ],
        "documentation": "/api/docs",
    }