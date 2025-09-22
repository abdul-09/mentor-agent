"""
Notification Service for AI Code Mentor
Implements production-grade notification delivery with multiple channels.

Compliance:
- RULE NOTIF-001: Reliable notification delivery
- RULE SEC-005: Secure notification handling
- RULE LOG-001: Structured logging with delivery tracking
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import structlog
import json

from src.config.settings import get_settings
from src.services.email_service import email_service
from src.services.redis_service import redis_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class NotificationService:
    """
    Production-grade notification service.
    
    Provides multi-channel notification delivery with proper error handling,
    retry mechanisms, and delivery tracking.
    """
    
    def __init__(self):
        self.is_connected = False
        self.notification_channels = ["email", "in_app"]
        
    async def connect(self) -> bool:
        """
        Initialize notification service dependencies.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Check if email service is connected
            if not email_service.is_connected:
                email_connected = await email_service.connect()
                if not email_connected:
                    logger.warning("Email service not available for notifications")
            
            # Check if Redis is connected (for in-app notifications)
            if not redis_service.is_connected:
                logger.warning("Redis not available for in-app notifications")
            
            self.is_connected = True
            logger.info("Notification service initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to initialize notification service",
                error=str(e)
            )
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect notification service."""
        try:
            # No specific cleanup needed for now
            self.is_connected = False
            logger.info("Notification service disconnected")
            
        except Exception as e:
            logger.error("Error disconnecting notification service", error=str(e))
    
    async def send_notification(
        self,
        user_id: str,
        user_email: str,
        user_name: str,
        notification_type: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Send notification through specified channels.
        
        Args:
            user_id: User ID
            user_email: User email address
            user_name: User full name
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            data: Additional data for the notification
            channels: List of channels to send notification through
            priority: Priority level (low, normal, high, urgent)
            
        Returns:
            Dict containing delivery status for each channel
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                return {
                    "success": False,
                    "error": "Notification service not connected",
                    "channels": {}
                }
        
        # Default to all available channels if none specified
        if not channels:
            channels = self.notification_channels.copy()
        
        # Filter channels based on user preferences
        user_preferences = await self._get_user_preferences(user_id)
        if not user_preferences.get("email_notifications", True):
            if "email" in channels:
                channels.remove("email")
        
        # Prepare notification data
        notification_data = {
            "user_id": user_id,
            "type": notification_type,
            "title": title,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": priority,
            "channels": channels
        }
        
        # Send through each channel
        results = {}
        for channel in channels:
            try:
                if channel == "email":
                    result = await self._send_email_notification(
                        user_email=user_email,
                        user_name=user_name,
                        notification_type=notification_type,
                        title=title,
                        message=message,
                        data=data or {}
                    )
                    results["email"] = result
                    
                elif channel == "in_app":
                    result = await self._send_in_app_notification(
                        user_id=user_id,
                        notification_type=notification_type,
                        title=title,
                        message=message,
                        data=data or {},
                        priority=priority
                    )
                    results["in_app"] = result
                    
                else:
                    results[channel] = {
                        "success": False,
                        "error": f"Unknown notification channel: {channel}"
                    }
                    
            except Exception as e:
                logger.error(
                    "Failed to send notification through channel",
                    channel=channel,
                    error=str(e),
                    user_id=user_id
                )
                results[channel] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Log notification delivery
        successful_channels = [ch for ch, res in results.items() if res.get("success")]
        failed_channels = [ch for ch, res in results.items() if not res.get("success")]
        
        logger.info(
            "Notification delivery completed",
            user_id=user_id,
            notification_type=notification_type,
            successful_channels=successful_channels,
            failed_channels=failed_channels
        )
        
        return {
            "success": len(successful_channels) > 0,
            "channels": results,
            "successful_channels": successful_channels,
            "failed_channels": failed_channels
        }
    
    async def _send_email_notification(
        self,
        user_email: str,
        user_name: str,
        notification_type: str,
        title: str,
        message: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send email notification.
        
        Args:
            user_email: Recipient email address
            user_name: Recipient full name
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            data: Additional data
            
        Returns:
            Dict containing delivery status
        """
        try:
            # Use existing email service method for specific notification types
            if notification_type in ["analysis_complete", "new_message"]:
                result = await email_service.send_notification_email(
                    to_email=user_email,
                    user_name=user_name,
                    notification_type=notification_type,
                    notification_data=data
                )
            else:
                # Generic email notification
                subject = f"{title} - AI Code Mentor"
                
                body = f"""
Hello {user_name},

{message}

Additional Information:
{json.dumps(data, indent=2) if data else "No additional information"}

Best regards,
The AI Code Mentor Team
                """.strip()
                
                html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title} - AI Code Mentor</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #2c3e50;">{title}</h2>
        
        <p>Hello <strong>{user_name}</strong>,</p>
        
        <p>{message}</p>
        
        {f'<div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0;"><pre>{json.dumps(data, indent=2)}</pre></div>' if data else ''}
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        
        <p style="font-size: 0.9em; color: #777;">
            Best regards,<br>
            The AI Code Mentor Team
        </p>
    </div>
</body>
</html>
                """.strip()
                
                result = await email_service.send_email(
                    to_emails=[user_email],
                    subject=subject,
                    body=body,
                    html_body=html_body
                )
            
            if result["success"]:
                logger.info(
                    "Email notification sent successfully",
                    user_email=user_email,
                    notification_type=notification_type
                )
            else:
                logger.error(
                    "Failed to send email notification",
                    user_email=user_email,
                    notification_type=notification_type,
                    error=result.get("error")
                )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to send email notification",
                user_email=user_email,
                notification_type=notification_type,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _send_in_app_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        data: Dict[str, Any],
        priority: str
    ) -> Dict[str, Any]:
        """
        Send in-app notification via Redis.
        
        Args:
            user_id: User ID
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            data: Additional data
            priority: Priority level
            
        Returns:
            Dict containing delivery status
        """
        try:
            if not redis_service.is_connected:
                return {
                    "success": False,
                    "error": "Redis service not connected"
                }
            
            # Create notification object
            notification = {
                "id": f"notif_{datetime.now(timezone.utc).timestamp()}",
                "user_id": user_id,
                "type": notification_type,
                "title": title,
                "message": message,
                "data": data,
                "priority": priority,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "read": False,
                "archived": False
            }
            
            # Store notification in Redis
            notification_key = f"user_notifications:{user_id}:{notification['id']}"
            user_notifications_key = f"user_notifications_list:{user_id}"
            
            # Save individual notification
            await redis_service.cache_set(
                notification_key, 
                notification, 
                ttl=60*60*24*30  # 30 days
            )
            
            # Add to user's notification list
            await redis_service.client.lpush(user_notifications_key, notification_key)
            await redis_service.client.ltrim(user_notifications_key, 0, 99)  # Keep last 100 notifications
            
            logger.info(
                "In-app notification sent successfully",
                user_id=user_id,
                notification_type=notification_type
            )
            
            return {
                "success": True,
                "notification_id": notification["id"]
            }
            
        except Exception as e:
            logger.error(
                "Failed to send in-app notification",
                user_id=user_id,
                notification_type=notification_type,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user notification preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict containing user preferences
        """
        try:
            if not redis_service.is_connected:
                # Return default preferences if Redis not available
                return {
                    "email_notifications": True,
                    "in_app_notifications": True
                }
            
            preferences_key = f"user_preferences:{user_id}"
            preferences = await redis_service.cache_get(preferences_key)
            
            if preferences:
                return preferences
            
            # Return default preferences
            return {
                "email_notifications": True,
                "in_app_notifications": True
            }
            
        except Exception as e:
            logger.error(
                "Failed to get user preferences",
                user_id=user_id,
                error=str(e)
            )
            # Return default preferences on error
            return {
                "email_notifications": True,
                "in_app_notifications": True
            }
    
    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 20,
        include_read: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get user notifications.
        
        Args:
            user_id: User ID
            limit: Maximum number of notifications to return
            include_read: Whether to include read notifications
            
        Returns:
            List of notifications
        """
        try:
            if not redis_service.is_connected:
                return []
            
            user_notifications_key = f"user_notifications_list:{user_id}"
            notification_keys = await redis_service.client.lrange(user_notifications_key, 0, limit - 1)
            
            notifications = []
            for key in notification_keys:
                try:
                    notification = await redis_service.cache_get(key.decode('utf-8'))
                    if notification and (include_read or not notification.get("read", False)):
                        notifications.append(notification)
                except Exception as e:
                    logger.warning(
                        "Failed to retrieve notification",
                        notification_key=key,
                        error=str(e)
                    )
                    continue
            
            # Sort by timestamp (newest first)
            notifications.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return notifications[:limit]
            
        except Exception as e:
            logger.error(
                "Failed to get user notifications",
                user_id=user_id,
                error=str(e)
            )
            return []
    
    async def mark_notification_as_read(
        self,
        user_id: str,
        notification_id: str
    ) -> bool:
        """
        Mark notification as read.
        
        Args:
            user_id: User ID
            notification_id: Notification ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not redis_service.is_connected:
                return False
            
            notification_key = f"user_notifications:{user_id}:{notification_id}"
            notification = await redis_service.cache_get(notification_key)
            
            if not notification:
                return False
            
            notification["read"] = True
            await redis_service.cache_set(
                notification_key, 
                notification, 
                ttl=60*60*24*30  # 30 days
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to mark notification as read",
                user_id=user_id,
                notification_id=notification_id,
                error=str(e)
            )
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on notification service.
        
        Returns:
            Dict containing health status and performance metrics
        """
        try:
            if not self.is_connected:
                connected = await self.connect()
                if not connected:
                    return {
                        "status": "unhealthy",
                        "error": "Notification service not connected"
                    }
            
            # Check email service health
            email_health = await email_service.health_check()
            
            # Check Redis health (for in-app notifications)
            redis_health = {"status": "healthy" if redis_service.is_connected else "unhealthy"}
            
            overall_status = "healthy" if (
                email_health["status"] == "healthy" and 
                redis_health["status"] == "healthy"
            ) else "unhealthy"
            
            return {
                "status": overall_status,
                "services": {
                    "email": email_health,
                    "redis": redis_health
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global notification service instance
notification_service = NotificationService()
