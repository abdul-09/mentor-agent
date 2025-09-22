"""
Email Service for AI Code Mentor
Implements production-grade email delivery with secure configuration.

Compliance:
- RULE SEC-005: Secure SMTP configuration
- RULE NOTIF-001: Reliable notification delivery
- RULE LOG-001: Structured logging with delivery tracking
"""

import asyncio
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Any
import structlog
import aiosmtplib
from aiosmtplib.errors import SMTPException

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class EmailService:
    """
    Production-grade asynchronous email service.
    
    Provides secure email delivery with proper error handling,
    retry mechanisms, and delivery tracking.
    """
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.smtp_use_tls = settings.SMTP_USE_TLS
        self.from_email = settings.SMTP_FROM_EMAIL
        self.from_name = settings.SMTP_FROM_NAME
        self.is_connected = False
        
        # Connection pool for async operations
        self._connection_pool = []
        
    async def connect(self) -> bool:
        """
        Test SMTP connection and credentials.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Test connection with aiosmtplib
            # For port 587, use STARTTLS; for port 465, use TLS; for port 25, use neither
            if self.smtp_port == 587:
                use_tls = False
                start_tls = True
            elif self.smtp_port == 465:
                use_tls = True
                start_tls = False
            else:  # port 25 or other
                use_tls = False
                start_tls = False
            
            await aiosmtplib.send(
                "Test connection",
                sender=self.from_email,
                recipients=[self.from_email],
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                use_tls=use_tls,
                start_tls=start_tls
            )
            
            self.is_connected = True
            logger.info(
                "Email service connected successfully",
                smtp_host=self.smtp_host,
                smtp_port=self.smtp_port
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to connect to email service",
                error=str(e),
                smtp_host=self.smtp_host,
                smtp_port=self.smtp_port
            )
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from SMTP server."""
        try:
            # Close any pooled connections
            for connection in self._connection_pool:
                if not connection.is_closed():
                    await connection.quit()
            
            self._connection_pool.clear()
            self.is_connected = False
            
            logger.info("Email service disconnected")
            
        except Exception as e:
            logger.error("Error disconnecting email service", error=str(e))
    
    async def send_email(
        self,
        to_emails: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        cc_emails: Optional[List[str]] = None,
        bcc_emails: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send email asynchronously with retry mechanism.
        
        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            body: Plain text email body
            html_body: Optional HTML email body
            attachments: Optional list of attachment dictionaries
            cc_emails: Optional CC recipients
            bcc_emails: Optional BCC recipients
            
        Returns:
            Dict containing delivery status and metadata
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                return {
                    "success": False,
                    "error": "Email service not connected",
                    "recipients": to_emails
                }
        
        # Prepare email message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self.from_name} <{self.from_email}>"
        message["To"] = ", ".join(to_emails)
        
        if cc_emails:
            message["Cc"] = ", ".join(cc_emails)
            
        # Add plain text body
        text_part = MIMEText(body, "plain")
        message.attach(text_part)
        
        # Add HTML body if provided
        if html_body:
            html_part = MIMEText(html_body, "html")
            message.attach(html_part)
        
        # Add attachments if provided
        if attachments:
            for attachment in attachments:
                part = MIMEBase('application', "octet-stream")
                part.set_payload(attachment.get("content"))
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment.get("filename")}"'
                )
                message.attach(part)
        
        # Combine all recipients
        all_recipients = to_emails.copy()
        if cc_emails:
            all_recipients.extend(cc_emails)
        if bcc_emails:
            all_recipients.extend(bcc_emails)
        
        # Send email with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # For port 587, use STARTTLS; for port 465, use TLS; for port 25, use neither
                if self.smtp_port == 587:
                    use_tls = False
                    start_tls = True
                elif self.smtp_port == 465:
                    use_tls = True
                    start_tls = False
                else:  # port 25 or other
                    use_tls = False
                    start_tls = False
                
                await aiosmtplib.send(
                    message=message,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.smtp_username,
                    password=self.smtp_password,
                    use_tls=use_tls,
                    start_tls=start_tls
                )
                
                logger.info(
                    "Email sent successfully",
                    subject=subject,
                    recipients=to_emails,
                    attempt=attempt + 1
                )
                
                return {
                    "success": True,
                    "message_id": "unknown",  # Would be available with proper SMTP library
                    "recipients": to_emails,
                    "attempt": attempt + 1
                }
                
            except SMTPException as e:
                logger.warning(
                    "SMTP error sending email",
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=max_retries
                )
                
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": f"SMTP error after {max_retries} attempts: {str(e)}",
                        "recipients": to_emails
                    }
                    
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(
                    "Unexpected error sending email",
                    error=str(e),
                    attempt=attempt + 1
                )
                
                return {
                    "success": False,
                    "error": f"Unexpected error: {str(e)}",
                    "recipients": to_emails
                }
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "recipients": to_emails
        }
    
    async def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        user_name: str,
        reset_link: str = None
    ) -> Dict[str, Any]:
        """
        Send password reset email to user.
        
        Args:
            to_email: Recipient email address
            reset_token: Password reset token
            user_name: User's full name
            reset_link: Optional direct reset link
            
        Returns:
            Dict containing delivery status
        """
        # Generate reset link if not provided
        if not reset_link:
            reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        
        # Email content
        subject = "Password Reset Request - AI Code Mentor"
        
        body = f"""
Hello {user_name},

You have requested a password reset for your AI Code Mentor account.

To reset your password, please click the link below:
{reset_link}

This link will expire in 1 hour.

If you did not request this password reset, please ignore this email or contact support if you have concerns.

Best regards,
The AI Code Mentor Team
        """.strip()
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Password Reset - AI Code Mentor</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #2c3e50;">Password Reset Request</h2>
        
        <p>Hello <strong>{user_name}</strong>,</p>
        
        <p>You have requested a password reset for your AI Code Mentor account.</p>
        
        <p style="margin: 30px 0;">
            <a href="{reset_link}" 
               style="background-color: #3498db; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 4px; display: inline-block;">
                Reset Password
            </a>
        </p>
        
        <p><strong>This link will expire in 1 hour.</strong></p>
        
        <p>If you did not request this password reset, please ignore this email or 
           <a href="mailto:{settings.SUPPORT_EMAIL}">contact support</a> if you have concerns.</p>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        
        <p style="font-size: 0.9em; color: #777;">
            Best regards,<br>
            The AI Code Mentor Team
        </p>
    </div>
</body>
</html>
        """.strip()
        
        # Send email
        result = await self.send_email(
            to_emails=[to_email],
            subject=subject,
            body=body,
            html_body=html_body
        )
        
        if result["success"]:
            logger.info(
                "Password reset email sent",
                user_email=to_email,
                user_name=user_name
            )
        else:
            logger.error(
                "Failed to send password reset email",
                user_email=to_email,
                error=result.get("error")
            )
        
        return result
    
    async def send_notification_email(
        self,
        to_email: str,
        user_name: str,
        notification_type: str,
        notification_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send notification email to user.
        
        Args:
            to_email: Recipient email address
            user_name: User's full name
            notification_type: Type of notification
            notification_data: Notification content data
            
        Returns:
            Dict containing delivery status
        """
        # Generate subject and content based on notification type
        if notification_type == "analysis_complete":
            subject = "Code Analysis Complete - AI Code Mentor"
            body = f"""
Hello {user_name},

Your code analysis has been completed successfully.

Analysis Details:
- Repository: {notification_data.get('repository_name', 'Unknown')}
- Files Analyzed: {notification_data.get('file_count', 0)}
- Issues Found: {notification_data.get('issue_count', 0)}

You can view the full analysis results in your dashboard.

Best regards,
The AI Code Mentor Team
            """.strip()
            
            html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Complete - AI Code Mentor</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #2c3e50;">Code Analysis Complete</h2>
        
        <p>Hello <strong>{user_name}</strong>,</p>
        
        <p>Your code analysis has been completed successfully.</p>
        
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0;">
            <h3 style="margin-top: 0;">Analysis Details</h3>
            <ul>
                <li><strong>Repository:</strong> {notification_data.get('repository_name', 'Unknown')}</li>
                <li><strong>Files Analyzed:</strong> {notification_data.get('file_count', 0)}</li>
                <li><strong>Issues Found:</strong> {notification_data.get('issue_count', 0)}</li>
            </ul>
        </div>
        
        <p>You can view the full analysis results in your <a href="{settings.FRONTEND_URL}/dashboard">dashboard</a>.</p>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        
        <p style="font-size: 0.9em; color: #777;">
            Best regards,<br>
            The AI Code Mentor Team
        </p>
    </div>
</body>
</html>
            """.strip()
            
        elif notification_type == "new_message":
            subject = "New Message - AI Code Mentor"
            body = f"""
Hello {user_name},

You have a new message in your conversation with AI Code Mentor.

Message: {notification_data.get('message_preview', '...')}

You can continue the conversation in your dashboard.

Best regards,
The AI Code Mentor Team
            """.strip()
            
            html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>New Message - AI Code Mentor</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #2c3e50;">New Message</h2>
        
        <p>Hello <strong>{user_name}</strong>,</p>
        
        <p>You have a new message in your conversation with AI Code Mentor.</p>
        
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0;">
            <p><strong>Message:</strong></p>
            <p>{notification_data.get('message_preview', '...')}</p>
        </div>
        
        <p>You can continue the conversation in your <a href="{settings.FRONTEND_URL}/chat">dashboard</a>.</p>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        
        <p style="font-size: 0.9em; color: #777;">
            Best regards,<br>
            The AI Code Mentor Team
        </p>
    </div>
</body>
</html>
            """.strip()
            
        else:
            subject = f"Notification - AI Code Mentor"
            body = f"""
Hello {user_name},

You have received a new notification from AI Code Mentor.

Notification: {notification_data.get('message', 'New notification')}

Best regards,
The AI Code Mentor Team
            """.strip()
            
            html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Notification - AI Code Mentor</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #2c3e50;">New Notification</h2>
        
        <p>Hello <strong>{user_name}</strong>,</p>
        
        <p>You have received a new notification from AI Code Mentor.</p>
        
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0;">
            <p>{notification_data.get('message', 'New notification')}</p>
        </div>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        
        <p style="font-size: 0.9em; color: #777;">
            Best regards,<br>
            The AI Code Mentor Team
        </p>
    </div>
</body>
</html>
            """.strip()
        
        # Send email
        result = await self.send_email(
            to_emails=[to_email],
            subject=subject,
            body=body,
            html_body=html_body
        )
        
        if result["success"]:
            logger.info(
                "Notification email sent",
                user_email=to_email,
                user_name=user_name,
                notification_type=notification_type
            )
        else:
            logger.error(
                "Failed to send notification email",
                user_email=to_email,
                notification_type=notification_type,
                error=result.get("error")
            )
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on email service.
        
        Returns:
            Dict containing health status and performance metrics
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.is_connected:
                connected = await self.connect()
                if not connected:
                    return {
                        "status": "unhealthy",
                        "error": "Not connected to SMTP server",
                        "response_time_ms": 0
                    }
            
            # Test with a simple connection test
            test_result = await self.send_email(
                to_emails=[self.from_email],
                subject="Health Check Test",
                body="This is an automated health check test."
            )
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if test_result["success"]:
                return {
                    "status": "healthy",
                    "smtp_host": self.smtp_host,
                    "smtp_port": self.smtp_port,
                    "response_time_ms": round(response_time, 2)
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": test_result.get("error"),
                    "response_time_ms": round(response_time, 2)
                }
                
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round(response_time, 2)
            }


# Global email service instance
email_service = EmailService()