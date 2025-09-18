"""
Structured Logging Configuration for AI Code Mentor
Implements RULE LOG-001 with structured JSON logging and trace IDs.

Compliance:
- RULE LOG-001: Structured logging format with trace IDs
- RULE LOG-002: What to log (authentication, errors, performance)
- RULE LOG-003: What NOT to log (passwords, PII, secrets)
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.processors import JSONRenderer
from structlog.stdlib import add_log_level, add_logger_name

from src.config.settings import get_settings

settings = get_settings()


def setup_logging() -> None:
    """
    Configure structured logging with JSON output.
    Implements RULE LOG-001 requirements.
    """
    # Configure structlog
    structlog.configure(
        processors=[
            # Add log level and logger name
            add_log_level,
            add_logger_name,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Stack info processor
            structlog.processors.StackInfoRenderer(),
            # Exception formatter
            structlog.dev.set_exc_info,
            # JSON renderer for structured output
            JSONRenderer() if settings.is_production else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )
    
    # Silence noisy loggers in production
    if settings.is_production:
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)


class SecurityAuditLogger:
    """
    Security-focused logging for audit trails.
    Implements RULE LOG-002 requirements.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("security.audit")
    
    def log_authentication_attempt(
        self,
        user_id: str = None,
        email: str = None,
        success: bool = False,
        reason: str = None,
        ip_address: str = None,
        user_agent: str = None,
        trace_id: str = None,
    ) -> None:
        """Log authentication attempts (RULE LOG-002)."""
        # Sanitize email for logging (show only domain)
        sanitized_email = None
        if email:
            sanitized_email = f"***@{email.split('@')[-1]}" if "@" in email else "***"
        
        self.logger.info(
            "Authentication attempt",
            event_type="auth_attempt",
            user_id=user_id,
            email=sanitized_email,
            success=success,
            failure_reason=reason if not success else None,
            client_ip=ip_address,
            user_agent=user_agent,
            trace_id=trace_id,
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        reason: str,
        ip_address: str = None,
        trace_id: str = None,
    ) -> None:
        """Log authorization failures (RULE LOG-002)."""
        self.logger.warning(
            "Authorization denied",
            event_type="auth_denied",
            user_id=user_id,
            resource=resource,
            action=action,
            reason=reason,
            client_ip=ip_address,
            trace_id=trace_id,
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: str = None,
        trace_id: str = None,
    ) -> None:
        """Log data access for audit trails (RULE PRIVACY-002)."""
        self.logger.info(
            "Data access",
            event_type="data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            client_ip=ip_address,
            trace_id=trace_id,
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        details: Dict[str, Any] = None,
        ip_address: str = None,
        user_id: str = None,
        trace_id: str = None,
    ) -> None:
        """Log security events and incidents."""
        self.logger.warning(
            "Security event",
            event_type="security_incident",
            incident_type=event_type,
            severity=severity,
            description=description,
            details=details or {},
            client_ip=ip_address,
            user_id=user_id,
            trace_id=trace_id,
        )


class PerformanceLogger:
    """
    Performance monitoring logger.
    Tracks metrics for RULE PERF-004 compliance.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("performance")
    
    def log_slow_query(
        self,
        query: str,
        duration_ms: float,
        table: str = None,
        user_id: str = None,
        trace_id: str = None,
    ) -> None:
        """Log slow database queries (>100ms)."""
        # Sanitize query for logging (remove sensitive data)
        sanitized_query = self._sanitize_query(query)
        
        self.logger.warning(
            "Slow database query",
            event_type="slow_query",
            query=sanitized_query,
            duration_ms=duration_ms,
            table=table,
            user_id=user_id,
            trace_id=trace_id,
        )
    
    def log_slow_api_response(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        status_code: int,
        user_id: str = None,
        trace_id: str = None,
    ) -> None:
        """Log slow API responses (>200ms)."""
        self.logger.warning(
            "Slow API response",
            event_type="slow_api",
            endpoint=endpoint,
            method=method,
            duration_ms=duration_ms,
            status_code=status_code,
            user_id=user_id,
            trace_id=trace_id,
        )
    
    def _sanitize_query(self, query: str) -> str:
        """Remove sensitive data from SQL queries."""
        # Replace potential sensitive values with placeholders
        import re
        
        # Remove string literals (potential sensitive data)
        sanitized = re.sub(r"'[^']*'", "'***'", query)
        # Remove numeric literals that might be IDs
        sanitized = re.sub(r'\b\d{4,}\b', '***', sanitized)
        
        return sanitized


class BusinessLogger:
    """
    Business event logger for tracking user actions.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("business")
    
    def log_file_upload(
        self,
        user_id: str,
        file_name: str,
        file_size: int,
        file_type: str,
        processing_time_ms: float,
        success: bool,
        trace_id: str = None,
    ) -> None:
        """Log file upload events."""
        # Sanitize filename (remove path, keep only extension)
        sanitized_name = f"file.{file_name.split('.')[-1]}" if "." in file_name else "file"
        
        self.logger.info(
            "File upload",
            event_type="file_upload",
            user_id=user_id,
            file_name=sanitized_name,
            file_size_bytes=file_size,
            file_type=file_type,
            processing_time_ms=processing_time_ms,
            success=success,
            trace_id=trace_id,
        )
    
    def log_ai_query(
        self,
        user_id: str,
        query_type: str,
        tokens_used: int,
        response_time_ms: float,
        success: bool,
        trace_id: str = None,
    ) -> None:
        """Log AI/LLM queries for usage tracking."""
        self.logger.info(
            "AI query",
            event_type="ai_query",
            user_id=user_id,
            query_type=query_type,
            tokens_used=tokens_used,
            response_time_ms=response_time_ms,
            success=success,
            trace_id=trace_id,
        )


# Global logger instances
security_logger = SecurityAuditLogger()
performance_logger = PerformanceLogger()
business_logger = BusinessLogger()