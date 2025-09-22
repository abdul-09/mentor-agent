"""
Multi-Factor Authentication Service for AI Code Mentor
Implements TOTP and backup codes for enhanced security.

Compliance:
- RULE AUTH-003: MFA for sensitive operations
- RULE SEC-005: Secure cryptographic operations
- RULE LOG-001: Security audit trails
"""

import base64
import secrets
import time
from typing import Dict, List, Optional, Tuple

import pyotp
import qrcode
import structlog
from io import BytesIO
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from src.config.settings import get_settings
from src.models.user import User

logger = structlog.get_logger(__name__)
settings = get_settings()


class MFAService:
    """
    Production-grade Multi-Factor Authentication service.
    
    Provides TOTP (Time-based One-Time Password) and backup codes
    with proper encryption and security practices.
    """
    
    def __init__(self):
        # Create encryption key from SECRET_KEY for secure storage
        self.cipher_suite = Fernet(
            base64.urlsafe_b64encode(settings.SECRET_KEY[:32].encode())
        )
        self.issuer_name = "AI Code Mentor"
        
    def _encrypt_secret(self, secret: str) -> str:
        """Encrypt MFA secret for secure storage."""
        try:
            return self.cipher_suite.encrypt(secret.encode()).decode()
        except Exception as e:
            logger.error("Failed to encrypt MFA secret", error=str(e))
            raise
    
    def _decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt MFA secret for verification."""
        try:
            return self.cipher_suite.decrypt(encrypted_secret.encode()).decode()
        except Exception as e:
            logger.error("Failed to decrypt MFA secret", error=str(e))
            raise
    
    def generate_totp_secret(self) -> str:
        """Generate a new TOTP secret key."""
        return pyotp.random_base32()
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """
        Generate backup codes for MFA recovery.
        
        Args:
            count: Number of backup codes to generate
            
        Returns:
            List of backup codes
        """
        backup_codes = []
        for _ in range(count):
            # Generate 8-digit backup codes
            code = ''.join([str(secrets.randbelow(10)) for _ in range(8)])
            backup_codes.append(code)
        
        return backup_codes
    
    def generate_qr_code(self, secret: str, user_email: str) -> str:
        """
        Generate QR code for TOTP setup.
        
        Args:
            secret: TOTP secret key
            user_email: User's email address
            
        Returns:
            Base64-encoded QR code image
        """
        try:
            # Create TOTP URI
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user_email,
                issuer_name=self.issuer_name
            )
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            # Create QR code image
            img = qr.make_image(fill_color="black", back_color="white")
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info(
                "QR code generated for MFA setup",
                user_email=user_email,
                qr_code_size=len(qr_code_base64)
            )
            
            return f"data:image/png;base64,{qr_code_base64}"
            
        except Exception as e:
            logger.error(
                "Failed to generate QR code",
                error=str(e),
                user_email=user_email
            )
            raise
    
    async def setup_mfa(
        self,
        user: User,
        db: AsyncSession
    ) -> Dict[str, any]:
        """
        Set up MFA for a user.
        
        Args:
            user: User instance
            db: Database session
            
        Returns:
            Dict containing setup information
        """
        try:
            # Generate TOTP secret
            secret = self.generate_totp_secret()
            
            # Generate backup codes
            backup_codes = self.generate_backup_codes()
            
            # Encrypt secret for storage
            encrypted_secret = self._encrypt_secret(secret)
            
            # Encrypt backup codes
            encrypted_backup_codes = [
                self._encrypt_secret(code) for code in backup_codes
            ]
            
            # Generate QR code
            qr_code_url = self.generate_qr_code(secret, user.email)
            
            # Update user record (but don't enable MFA yet)
            user.mfa_secret = encrypted_secret
            user.mfa_backup_codes = encrypted_backup_codes
            # MFA is enabled only after successful verification
            
            await db.commit()
            
            logger.info(
                "MFA setup completed",
                user_id=str(user.id),
                user_email=user.email,
                backup_codes_count=len(backup_codes)
            )
            
            return {
                "qr_code_url": qr_code_url,
                "secret_key": secret,  # Return for manual entry
                "backup_codes": backup_codes
            }
            
        except Exception as e:
            logger.error(
                "MFA setup failed",
                user_id=str(user.id),
                error=str(e)
            )
            await db.rollback()
            raise
    
    def verify_totp_code(
        self,
        secret: str,
        provided_code: str,
        valid_window: int = 1
    ) -> bool:
        """
        Verify TOTP code.
        
        Args:
            secret: TOTP secret key
            provided_code: Code provided by user
            valid_window: Number of time windows to check (for clock drift)
            
        Returns:
            bool: True if code is valid
        """
        try:
            totp = pyotp.TOTP(secret)
            
            # Check current time window and adjacent windows for clock drift
            current_time = int(time.time())
            
            for window_offset in range(-valid_window, valid_window + 1):
                window_time = current_time + (window_offset * 30)  # 30-second windows
                expected_code = totp.at(window_time)
                
                if provided_code == expected_code:
                    logger.debug(
                        "TOTP code verified successfully",
                        window_offset=window_offset
                    )
                    return True
            
            logger.warning(
                "TOTP code verification failed",
                provided_code_length=len(provided_code)
            )
            return False
            
        except Exception as e:
            logger.error(
                "TOTP verification error",
                error=str(e)
            )
            return False
    
    def verify_backup_code(
        self,
        encrypted_backup_codes: List[str],
        provided_code: str
    ) -> Tuple[bool, List[str]]:
        """
        Verify backup code and remove it from the list.
        
        Args:
            encrypted_backup_codes: List of encrypted backup codes
            provided_code: Code provided by user
            
        Returns:
            Tuple of (is_valid, updated_backup_codes_list)
        """
        try:
            updated_codes = encrypted_backup_codes.copy()
            
            for i, encrypted_code in enumerate(encrypted_backup_codes):
                try:
                    decrypted_code = self._decrypt_secret(encrypted_code)
                    
                    if provided_code == decrypted_code:
                        # Remove used backup code
                        updated_codes.pop(i)
                        
                        logger.info(
                            "Backup code verified and consumed",
                            remaining_codes=len(updated_codes)
                        )
                        
                        return True, updated_codes
                        
                except Exception as e:
                    logger.warning(
                        "Failed to decrypt backup code",
                        code_index=i,
                        error=str(e)
                    )
                    continue
            
            logger.warning(
                "Backup code verification failed",
                provided_code_length=len(provided_code)
            )
            return False, encrypted_backup_codes
            
        except Exception as e:
            logger.error(
                "Backup code verification error",
                error=str(e)
            )
            return False, encrypted_backup_codes
    
    async def verify_mfa_code(
        self,
        user: User,
        provided_code: str,
        db: AsyncSession
    ) -> Dict[str, any]:
        """
        Verify MFA code (TOTP or backup code).
        
        Args:
            user: User instance
            provided_code: Code provided by user
            db: Database session
            
        Returns:
            Dict containing verification result
        """
        try:
            if not user.mfa_secret:
                return {
                    "verified": False,
                    "error": "MFA not set up for user"
                }
            
            # Decrypt TOTP secret
            try:
                secret = self._decrypt_secret(user.mfa_secret)
            except Exception as e:
                logger.error(
                    "Failed to decrypt MFA secret",
                    user_id=str(user.id),
                    error=str(e)
                )
                return {
                    "verified": False,
                    "error": "MFA configuration error"
                }
            
            # First try TOTP verification
            if self.verify_totp_code(secret, provided_code):
                # Enable MFA if this is the first successful verification
                if not user.mfa_enabled:
                    user.mfa_enabled = True
                    await db.commit()
                    
                    logger.info(
                        "MFA enabled for user after first successful verification",
                        user_id=str(user.id)
                    )
                
                return {
                    "verified": True,
                    "method": "totp",
                    "mfa_enabled": user.mfa_enabled
                }
            
            # If TOTP fails, try backup codes
            if user.mfa_backup_codes:
                is_valid, updated_codes = self.verify_backup_code(
                    user.mfa_backup_codes,
                    provided_code
                )
                
                if is_valid:
                    # Update backup codes in database
                    user.mfa_backup_codes = updated_codes
                    
                    # Enable MFA if this is the first successful verification
                    if not user.mfa_enabled:
                        user.mfa_enabled = True
                    
                    await db.commit()
                    
                    logger.info(
                        "MFA verified with backup code",
                        user_id=str(user.id),
                        remaining_backup_codes=len(updated_codes)
                    )
                    
                    return {
                        "verified": True,
                        "method": "backup_code",
                        "remaining_backup_codes": len(updated_codes),
                        "mfa_enabled": user.mfa_enabled
                    }
            
            # Both TOTP and backup code verification failed
            logger.warning(
                "MFA verification failed for user",
                user_id=str(user.id),
                provided_code_length=len(provided_code)
            )
            
            return {
                "verified": False,
                "error": "Invalid MFA code"
            }
            
        except Exception as e:
            logger.error(
                "MFA verification error",
                user_id=str(user.id),
                error=str(e)
            )
            return {
                "verified": False,
                "error": "MFA verification failed"
            }
    
    async def disable_mfa(
        self,
        user: User,
        db: AsyncSession
    ) -> bool:
        """
        Disable MFA for a user.
        
        Args:
            user: User instance
            db: Database session
            
        Returns:
            bool: True if successful
        """
        try:
            user.mfa_enabled = False
            user.mfa_secret = None
            user.mfa_backup_codes = None
            
            await db.commit()
            
            logger.info(
                "MFA disabled for user",
                user_id=str(user.id)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to disable MFA",
                user_id=str(user.id),
                error=str(e)
            )
            await db.rollback()
            return False
    
    async def regenerate_backup_codes(
        self,
        user: User,
        db: AsyncSession
    ) -> Optional[List[str]]:
        """
        Regenerate backup codes for a user.
        
        Args:
            user: User instance
            db: Database session
            
        Returns:
            List of new backup codes or None if failed
        """
        try:
            if not user.mfa_enabled:
                logger.warning(
                    "Attempted to regenerate backup codes for user without MFA",
                    user_id=str(user.id)
                )
                return None
            
            # Generate new backup codes
            backup_codes = self.generate_backup_codes()
            
            # Encrypt backup codes
            encrypted_backup_codes = [
                self._encrypt_secret(code) for code in backup_codes
            ]
            
            # Update user record
            user.mfa_backup_codes = encrypted_backup_codes
            await db.commit()
            
            logger.info(
                "Backup codes regenerated",
                user_id=str(user.id),
                new_code_count=len(backup_codes)
            )
            
            return backup_codes
            
        except Exception as e:
            logger.error(
                "Failed to regenerate backup codes",
                user_id=str(user.id),
                error=str(e)
            )
            await db.rollback()
            return None


# Global MFA service instance
mfa_service = MFAService()