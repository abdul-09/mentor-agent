# AI Code Mentor API Documentation

This document provides an overview of the backend API structure for the AI Code Mentor application.

## API Versioning

All API endpoints are prefixed with `/api/v1/` to indicate version 1 of the API.

## Authentication Endpoints

Base path: `/api/v1/auth`

### Register
- **Endpoint**: `POST /api/v1/auth/register`
- **Description**: Register a new user account
- **Request Body**:
  ```json
  {
    "email": "string",
    "password": "string",
    "full_name": "string",
    "accept_terms": "boolean"
  }
  ```
- **Response**:
  ```json
  {
    "access_token": "string",
    "refresh_token": "string",
    "token_type": "string",
    "expires_in": "integer",
    "user_id": "string"
  }
  ```

### Login
- **Endpoint**: `POST /api/v1/auth/login`
- **Description**: Authenticate user and return JWT tokens
- **Request Body**:
  ```json
  {
    "email": "string",
    "password": "string",
    "remember_me": "boolean"
  }
  ```
- **Response**:
  ```json
  {
    "access_token": "string",
    "refresh_token": "string",
    "token_type": "string",
    "expires_in": "integer",
    "user_id": "string"
  }
  ```

### Refresh Token
- **Endpoint**: `POST /api/v1/auth/refresh`
- **Description**: Refresh access token using refresh token
- **Request Body**:
  ```json
  {
    "refresh_token": "string"
  }
  ```
- **Response**:
  ```json
  {
    "access_token": "string",
    "refresh_token": "string",
    "token_type": "string",
    "expires_in": "integer",
    "user_id": "string"
  }
  ```

### Logout
- **Endpoint**: `POST /api/v1/auth/logout`
- **Description**: Logout user and invalidate tokens
- **Response**: `204 No Content`

### Password Reset Request
- **Endpoint**: `POST /api/v1/auth/password-reset`
- **Description**: Request password reset email
- **Request Body**:
  ```json
  {
    "email": "string"
  }
  ```
- **Response**:
  ```json
  {
    "message": "string"
  }
  ```

### Password Reset Confirmation
- **Endpoint**: `POST /api/v1/auth/password-reset/confirm`
- **Description**: Confirm password reset with new password
- **Request Body**:
  ```json
  {
    "token": "string",
    "new_password": "string"
  }
  ```
- **Response**:
  ```json
  {
    "message": "string"
  }
  ```

### MFA Setup
- **Endpoint**: `POST /api/v1/auth/mfa/setup`
- **Description**: Set up Multi-Factor Authentication
- **Response**:
  ```json
  {
    "qr_code_url": "string",
    "secret_key": "string",
    "backup_codes": ["string"]
  }
  ```

### MFA Verify
- **Endpoint**: `POST /api/v1/auth/mfa/verify`
- **Description**: Verify MFA code and complete authentication
- **Request Body**:
  ```json
  {
    "code": "string"
  }
  ```
- **Response**:
  ```json
  {
    "message": "string"
  }
  ```

### MFA Disable
- **Endpoint**: `POST /api/v1/auth/mfa/disable`
- **Description**: Disable Multi-Factor Authentication
- **Response**:
  ```json
  {
    "message": "string"
  }
  ```

### Regenerate MFA Backup Codes
- **Endpoint**: `POST /api/v1/auth/mfa/regenerate-backup-codes`
- **Description**: Regenerate backup codes for MFA
- **Response**:
  ```json
  {
    "message": "string",
    "backup_codes": ["string"]
  }
  ```

## User Endpoints

Base path: `/api/v1/users`

### Get User Profile
- **Endpoint**: `GET /api/v1/users/profile`
- **Description**: Get current user's profile information
- **Response**:
  ```json
  {
    "id": "string",
    "email": "string",
    "full_name": "string",
    "is_active": "boolean",
    "is_verified": "boolean",
    "is_admin": "boolean",
    "mfa_enabled": "boolean",
    "created_at": "string",
    "last_login_at": "string"
  }
  ```

### Update User Profile
- **Endpoint**: `PUT /api/v1/users/profile`
- **Description**: Update current user's profile information
- **Request Body**:
  ```json
  {
    "full_name": "string",
    "email": "string"
  }
  ```
- **Response**:
  ```json
  {
    "id": "string",
    "email": "string",
    "full_name": "string",
    "is_active": "boolean",
    "is_verified": "boolean",
    "is_admin": "boolean",
    "mfa_enabled": "boolean",
    "created_at": "string",
    "last_login_at": "string"
  }
  ```

## File Management Endpoints

Base path: `/api/v1/files`

### Upload File
- **Endpoint**: `POST /api/v1/files/upload`
- **Description**: Upload a new file
- **Request**: Multipart form data with file
- **Response**:
  ```json
  {
    "id": "string",
    "original_filename": "string",
    "file_size": "integer",
    "file_type": "string",
    "upload_status": "string",
    "created_at": "string"
  }
  ```

### List Files
- **Endpoint**: `GET /api/v1/files`
- **Description**: List user's files with pagination
- **Query Parameters**:
  - `limit`: integer (default: 50)
  - `offset`: integer (default: 0)
  - `file_type`: string (optional)
- **Response**:
  ```json
  {
    "files": [
      {
        "id": "string",
        "original_filename": "string",
        "file_size": "integer",
        "file_type": "string",
        "upload_status": "string",
        "created_at": "string"
      }
    ],
    "total": "integer",
    "limit": "integer",
    "offset": "integer"
  }
  ```

### Get File Details
- **Endpoint**: `GET /api/v1/files/{file_id}`
- **Description**: Get detailed information about a specific file
- **Response**:
  ```json
  {
    "id": "string",
    "original_filename": "string",
    "file_size": "integer",
    "file_type": "string",
    "upload_status": "string",
    "created_at": "string"
  }
  ```

### Delete File
- **Endpoint**: `DELETE /api/v1/files/{file_id}`
- **Description**: Delete a file
- **Response**: `204 No Content`

## Analysis Endpoints

Base path: `/api/v1/analysis`

### Analyze PDF
- **Endpoint**: `POST /api/v1/analysis/pdf`
- **Description**: Analyze uploaded PDF and extract content for Q&A
- **Request Body**:
  ```json
  {
    "file_id": "string",
    "analysis_type": "string",
    "include_embeddings": "boolean"
  }
  ```
- **Response**:
  ```json
  {
    "id": "string",
    "analysis_type": "string",
    "status": "string",
    "result": {},
    "error_message": "string",
    "created_at": "string",
    "updated_at": "string",
    "metadata": {}
  }
  ```

### Analyze GitHub Repository
- **Endpoint**: `POST /api/v1/analysis/github`
- **Description**: Analyze GitHub repository structure and code
- **Request Body**:
  ```json
  {
    "repository_url": "string",
    "branch": "string",
    "include_dependencies": "boolean",
    "language_filter": ["string"]
  }
  ```
- **Response**:
  ```json
  {
    "id": "string",
    "analysis_type": "string",
    "status": "string",
    "result": {},
    "error_message": "string",
    "created_at": "string",
    "updated_at": "string",
    "metadata": {}
  }
  ```

### Ask Question (Q&A)
- **Endpoint**: `POST /api/v1/analysis/qa`
- **Description**: Ask questions about analyzed content using AI
- **Request Body**:
  ```json
  {
    "question": "string",
    "analysis_id": "string",
    "conversation_id": "string"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "string",
    "confidence": "number",
    "sources": ["string"],
    "conversation_id": "string",
    "timestamp": "string"
  }
  ```

### List Analysis Sessions
- **Endpoint**: `GET /api/v1/analysis/sessions`
- **Description**: List user's analysis sessions with optional filtering
- **Query Parameters**:
  - `limit`: integer (default: 50)
  - `offset`: integer (default: 0)
  - `analysis_type`: string (optional)
  - `status`: string (optional)
- **Response**:
  ```json
  {
    "analyses": [
      {
        "id": "string",
        "analysis_type": "string",
        "status": "string",
        "result": {},
        "error_message": "string",
        "created_at": "string",
        "updated_at": "string",
        "metadata": {}
      }
    ],
    "total": "integer",
    "limit": "integer",
    "offset": "integer"
  }
  ```

### Get Analysis Details
- **Endpoint**: `GET /api/v1/analysis/{analysis_id}`
- **Description**: Get detailed information about a specific analysis
- **Response**:
  ```json
  {
    "id": "string",
    "analysis_type": "string",
    "status": "string",
    "result": {},
    "error_message": "string",
    "created_at": "string",
    "updated_at": "string",
    "metadata": {}
  }
  ```

### List Conversations
- **Endpoint**: `GET /api/v1/analysis/conversations`
- **Description**: List user's conversation history
- **Query Parameters**:
  - `limit`: integer (default: 20)
  - `offset`: integer (default: 0)
- **Response**:
  ```json
  {
    "conversations": [
      {
        "conversation_id": "string",
        "title": "string",
        "message_count": "integer",
        "last_activity": "string",
        "related_analysis_ids": ["string"]
      }
    ],
    "total": "integer",
    "limit": "integer",
    "offset": "integer"
  }
  ```

### Get Conversation History
- **Endpoint**: `GET /api/v1/analysis/conversations/{conversation_id}`
- **Description**: Get detailed conversation history
- **Query Parameters**:
  - `limit`: integer (default: 50)
- **Response**:
  ```json
  {
    "conversation_id": "string",
    "messages": [
      {
        "role": "string",
        "content": "string",
        "timestamp": "string",
        "analysis_id": "string",
        "sources": ["string"]
      }
    ],
    "total_messages": "integer",
    "created_at": "string",
    "last_activity": "string"
  }
  ```

### Delete Conversation
- **Endpoint**: `DELETE /api/v1/analysis/conversations/{conversation_id}`
- **Description**: Delete a conversation and all its messages
- **Response**: `204 No Content`

## Notification Endpoints

Base path: `/api/v1/notifications`

### List Notifications
- **Endpoint**: `GET /api/v1/notifications`
- **Description**: List user's notifications with pagination
- **Query Parameters**:
  - `limit`: integer (default: 50)
  - `offset`: integer (default: 0)
  - `unread_only`: boolean (default: false)
- **Response**:
  ```json
  {
    "notifications": [
      {
        "id": "string",
        "type": "string",
        "title": "string",
        "message": "string",
        "is_read": "boolean",
        "created_at": "string"
      }
    ],
    "total": "integer",
    "limit": "integer",
    "offset": "integer"
  }
  ```

### Mark Notification as Read
- **Endpoint**: `PUT /api/v1/notifications/{notification_id}/read`
- **Description**: Mark a notification as read
- **Response**:
  ```json
  {
    "id": "string",
    "type": "string",
    "title": "string",
    "message": "string",
    "is_read": "boolean",
    "created_at": "string"
  }
  ```

### Mark All Notifications as Read
- **Endpoint**: `POST /api/v1/notifications/read-all`
- **Description**: Mark all notifications as read
- **Response**:
  ```json
  {
    "message": "string"
  }
  ```

### Delete Notification
- **Endpoint**: `DELETE /api/v1/notifications/{notification_id}`
- **Description**: Delete a notification
- **Response**: `204 No Content`

## Advanced Analysis Endpoints

Base path: `/api/v1/advanced-analysis`

### Performance Analysis
- **Endpoint**: `POST /api/v1/advanced-analysis/performance`
- **Description**: Analyze code for performance issues and complexity
- **Request Body**:
  ```json
  {
    "code": "string",
    "language": "string"
  }
  ```
- **Response**:
  ```json
  {
    "issues": [
      {
        "type": "string",
        "description": "string",
        "location": "string",
        "complexity": "string",
        "recommendation": "string"
      }
    ]
  }
  ```

### Pattern Recognition
- **Endpoint**: `POST /api/v1/advanced-analysis/patterns`
- **Description**: Identify design patterns in code
- **Request Body**:
  ```json
  {
    "code": "string",
    "language": "string"
  }
  ```
- **Response**:
  ```json
  {
    "patterns": [
      {
        "name": "string",
        "description": "string",
        "location": "string",
        "benefits": ["string"],
        "drawbacks": ["string"]
      }
    ]
  }
  ```

### Consistency Check
- **Endpoint**: `POST /api/v1/advanced-analysis/consistency`
- **Description**: Check code consistency across a codebase
- **Request Body**:
  ```json
  {
    "files": [
      {
        "name": "string",
        "content": "string",
        "language": "string"
      }
    ]
  }
  ```
- **Response**:
  ```json
  {
    "inconsistencies": [
      {
        "type": "string",
        "description": "string",
        "files": ["string"],
        "recommendation": "string"
      }
    ]
  }
  ```

### Alternative Suggestions
- **Endpoint**: `POST /api/v1/advanced-analysis/suggestions`
- **Description**: Provide alternative implementations with trade-off analysis
- **Request Body**:
  ```json
  {
    "code": "string",
    "language": "string",
    "context": "string"
  }
  ```
- **Response**:
  ```json
  {
    "suggestions": [
      {
        "title": "string",
        "description": "string",
        "code": "string",
        "pros": ["string"],
        "cons": ["string"],
        "complexity": "string"
      }
    ]
  }
  ```

## GitHub Analysis Endpoints

Base path: `/api/v1/github`

### Analyze Repository
- **Endpoint**: `POST /api/v1/github/analyze`
- **Description**: Analyze a GitHub repository
- **Request Body**:
  ```json
  {
    "repository_url": "string",
    "branch": "string"
  }
  ```
- **Response**:
  ```json
  {
    "repository": {
      "name": "string",
      "url": "string",
      "branch": "string",
      "languages": ["string"],
      "files": "integer",
      "last_commit": "string"
    },
    "analysis": {
      "architecture": {},
      "dependencies": {},
      "issues": []
    }
  }
  ```

## Admin Endpoints

Base path: `/api/v1/admin`

### Health Check
- **Endpoint**: `GET /api/v1/admin/health`
- **Description**: Check system health and service status
- **Response**:
  ```json
  {
    "status": "string",
    "services": [
      {
        "name": "string",
        "status": "string",
        "response_time_ms": "number"
      }
    ],
    "timestamp": "string"
  }
  ```

### System Stats
- **Endpoint**: `GET /api/v1/admin/stats`
- **Description**: Get system statistics
- **Response**:
  ```json
  {
    "users": "integer",
    "analyses": "integer",
    "files": "integer",
    "uptime": "string"
  }
  ```