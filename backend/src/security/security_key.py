import secrets
import base64

# Generate a secure random secret key (64 characters)
secret_key = secrets.token_hex(32)
print(f"SECRET_KEY={secret_key}")

# Generate a secure JWT secret key (86 characters)
jwt_secret = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode('utf-8')
print(f"JWT_SECRET_KEY={jwt_secret}")