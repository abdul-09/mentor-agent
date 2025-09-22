"""
AI Code Mentor - Production-Grade Configuration Settings
Follows security best practices for environment separation and secret management.

Compliance:
- RULE SEC-004: Environment separation (dev/staging/production)
- RULE SEC-005: Secret management with secure defaults
- RULE PERF-002: Database connection configuration
- RULE AUTH-002: JWT token configuration
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with security and performance optimizations."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application Settings
    PROJECT_NAME: str = "AI Code Mentor"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = Field(default=False)
    
    # Server Configuration
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000, ge=1, le=65535)
    
    # Security Settings (RULE SEC-002, SEC-004)
    SECRET_KEY: str = Field(min_length=32)
    ALLOWED_HOSTS: List[str] = Field(default=["localhost", "127.0.0.1"])
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000"])
    
    # JWT Configuration (RULE AUTH-002)
    JWT_SECRET_KEY: str = Field(min_length=32)
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, le=15)  # Max 15 minutes
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, le=7)      # Max 7 days
    
    # Database Configuration (RULE DB-001, PERF-002)
    DATABASE_URL: str = Field(...)
    DB_POOL_SIZE: int = Field(default=5, ge=5, le=20)
    DB_MAX_OVERFLOW: int = Field(default=10)
    DB_POOL_TIMEOUT: int = Field(default=30)
    DB_POOL_RECYCLE: int = Field(default=3600)
    
    # Redis Configuration (RULE PERF-003, AUTH-004)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_MAX_CONNECTIONS: int = Field(default=20)
    
    # OpenAI Configuration (RULE SEC-005)
    OPENAI_API_KEY: str = Field(...)
    OPENAI_ORG_ID: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002")
    OPENAI_MAX_TOKENS: int = Field(default=4000)
    OPENAI_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = Field(...)
    PINECONE_ENVIRONMENT: str = Field(...)
    PINECONE_INDEX_NAME: str = Field(default="ai-code-mentor")
    
    # GitHub Configuration
    GITHUB_TOKEN: Optional[str] = Field(default=None)
    GITHUB_CLIENT_ID: Optional[str] = Field(default=None)
    GITHUB_CLIENT_SECRET: Optional[str] = Field(default=None)
    GITHUB_ACCESS_TOKEN: Optional[str] = Field(default=None)
    
    # File Upload Configuration (RULE SEC-001)
    MAX_FILE_SIZE_MB: int = Field(default=50, le=100)
    ALLOWED_FILE_TYPES: List[str] = Field(default=["application/pdf"])
    UPLOAD_DIR: str = Field(default="uploads")
    
    # Rate Limiting (RULE PERF-005)
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60)
    RATE_LIMIT_BURST: int = Field(default=10)
    
    # Monitoring & Logging (RULE MON-001, LOG-001)
    SENTRY_DSN: Optional[str] = Field(default=None)
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1, ge=0.0, le=1.0)
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Health Check Configuration
    HEALTH_CHECK_TIMEOUT: int = Field(default=30)
    
    # Email Configuration (RULE SEC-005)
    SMTP_HOST: str = Field(default="smtp.gmail.com")
    SMTP_PORT: int = Field(default=587)
    SMTP_USERNAME: str = Field(...)
    SMTP_PASSWORD: str = Field(...)
    SMTP_USE_TLS: bool = Field(default=True)
    SMTP_FROM_EMAIL: str = Field(...)
    SMTP_FROM_NAME: str = Field(default="AI Code Mentor")
    FRONTEND_URL: str = Field(default="http://localhost:3000")
    SUPPORT_EMAIL: str = Field(default="support@ai-code-mentor.com")
    
    # Testing Configuration
    TESTING: bool = Field(default=False)
    TEST_DATABASE_URL: Optional[str] = Field(default=None)
    
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be one of: development, staging, production")
        return v
    
    @field_validator("SECRET_KEY", "JWT_SECRET_KEY")
    @classmethod
    def validate_secret_keys(cls, v: str) -> str:
        """Validate secret keys are sufficiently long."""
        if len(v) < 32:
            raise ValueError("Secret keys must be at least 32 characters long")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == "development"
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL."""
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


class ProductionSettings(Settings):
    """Production-specific settings with enhanced security."""
    
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    
    # Enhanced security for production
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 10  # Shorter tokens in production
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 30   # Stricter rate limiting
    
    @field_validator("ALLOWED_HOSTS")
    @classmethod
    def validate_production_hosts(cls, v: List[str]) -> List[str]:
        """Ensure production hosts are properly configured."""
        if "localhost" in v or "127.0.0.1" in v:
            raise ValueError("Production cannot use localhost in ALLOWED_HOSTS")
        return v


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # More permissive for development
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 120


class TestingSettings(Settings):
    """Testing-specific settings."""
    
    TESTING: bool = True
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Use in-memory databases for testing
    DATABASE_URL: str = "sqlite+aiosqlite:///:memory:"
    REDIS_URL: str = "redis://localhost:6379/15"  # Separate Redis DB for tests


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance based on environment.
    
    Returns:
        Settings: Configured settings instance
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export settings instance
settings = get_settings()