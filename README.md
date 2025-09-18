# AI Code Mentor Backend

A production-grade FastAPI backend for the AI Code Mentor platform, implementing enterprise-level security, performance, and operational excellence standards.

## 🏗️ Architecture

This backend follows production-grade architecture principles with:

- **Security First**: Implementation of all security rules (AUTH-001, SEC-001, SEC-002)
- **Performance Optimized**: Sub-200ms API responses, connection pooling, caching
- **Production Ready**: Health checks, monitoring, structured logging, CI/CD ready
- **Scalable Design**: Microservice-ready architecture with clear separation of concerns

## 📁 Project Structure

```
backend/
├── src/
│   ├── api/                 # API endpoints (versioned under /api/v1/)
│   │   └── v1/
│   │       ├── endpoints/   # Route handlers
│   │       └── router.py    # API router configuration
│   ├── auth/               # Authentication & authorization
│   ├── config/             # Configuration management
│   ├── models/             # Database models
│   ├── security/           # Security middleware & utilities
│   ├── services/           # Business logic
│   └── utils/              # Utility functions
├── tests/                  # Test files (80% coverage requirement)
├── docs/                   # API documentation
├── scripts/                # Deployment scripts
├── migrations/             # Database migrations
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── Dockerfile             # Container configuration
```

## 🔐 Security Features

### Authentication & Authorization
- **bcrypt** password hashing (cost factor 12+)
- **JWT tokens** with 15-minute access and 7-day refresh cycles
- **Multi-Factor Authentication** (TOTP + backup codes)
- **Session management** with Redis encryption
- **Rate limiting** and brute force protection

### API Security
- **Security headers** (HSTS, CSP, XSS protection)
- **Input validation** and sanitization
- **SQL injection** prevention
- **CORS** configuration
- **Request size** limits

## ⚡ Performance Features

### Response Time SLAs
- API endpoints: <200ms (95th percentile)
- Database queries: <100ms (95th percentile)
- File uploads: <5s for 10MB files

### Optimization
- **Connection pooling** (PostgreSQL & Redis)
- **Multi-tier caching** (application, Redis, CDN)
- **Compression** for responses >1KB
- **Pagination** (max 100 items per request)

## 📊 Monitoring & Observability

### Structured Logging
- **JSON format** with trace IDs
- **Security audit** trails
- **Performance** metrics
- **Business event** tracking

### Health Checks
- **Database** connectivity
- **Redis** performance
- **External APIs** status
- **System resources** monitoring

### Metrics & Alerting
- **Prometheus** metrics collection
- **Grafana** dashboards
- **Sentry** error tracking
- **Real-time alerts** for critical issues

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Local Development Setup

1. **Clone and setup environment:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Environment configuration:**
```bash
cp .env.example .env
# Edit .env with your actual values (API keys, database URLs, etc.)
```

3. **Database setup:**
```bash
# Start PostgreSQL and Redis
docker-compose up postgres redis -d

# Run database migrations
alembic upgrade head
```

4. **Start the development server:**
```bash
python main.py
# OR
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

### API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## 🔧 Configuration

### Environment Variables

Key configuration options (see `.env.example` for complete list):

```bash
# Security (REQUIRED - Generate secure values!)
SECRET_KEY="your-super-secure-secret-key-min-32-chars"
JWT_SECRET_KEY="your-jwt-secret-key-min-32-chars"

# Database
DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/ai_code_mentor"

# AI Services (REQUIRED)
OPENAI_API_KEY="your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_ENVIRONMENT="your-pinecone-environment"

# Monitoring (Optional)
SENTRY_DSN="your-sentry-dsn"
```

### Production Settings

For production deployment:

```bash
ENVIRONMENT="production"
DEBUG=false
LOG_LEVEL="WARNING"
```

## 🧪 Testing

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/security/      # Security tests

# Performance testing
pytest tests/performance/   # Load tests
```

## 📈 Performance Monitoring

### Key Metrics
- **Response Time**: 95th percentile <200ms
- **Error Rate**: <1% for critical paths
- **Uptime**: >99.5%
- **Resource Usage**: CPU <70%, Memory <80%

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Sentry**: Error tracking and alerting
- **Structured Logs**: JSON format with trace IDs

## 🔄 Development Workflow

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Security scan
bandit -r src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 🚀 Deployment

### Production Checklist
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup procedures tested
- [ ] Security scan completed
- [ ] Load testing passed

### Docker Production
```bash
# Build production image
docker build -t ai-code-mentor-backend .

# Run with production settings
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=your-prod-db-url \
  ai-code-mentor-backend
```

## 📋 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Token refresh
- `POST /api/v1/auth/logout` - User logout
- `POST /api/v1/auth/mfa/setup` - MFA setup
- `POST /api/v1/auth/mfa/verify` - MFA verification

### File Management
- `GET /api/v1/files/` - List files
- `POST /api/v1/files/upload` - Upload PDF
- `GET /api/v1/files/{id}` - Get file
- `DELETE /api/v1/files/{id}` - Delete file

### Analysis
- `POST /api/v1/analysis/pdf` - Analyze PDF
- `POST /api/v1/analysis/github` - Analyze repository
- `POST /api/v1/analysis/qa` - Ask questions
- `GET /api/v1/analysis/sessions` - List sessions

### System
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## 🔍 Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check database is running
docker-compose ps postgres

# Check connection string
echo $DATABASE_URL

# Test connection
python -c "import asyncpg; print('OK')"
```

**Redis Connection Failed**
```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
redis-cli ping
```

**High Response Times**
```bash
# Check database query performance
docker-compose logs postgres

# Monitor resource usage
docker stats

# Check application logs
docker-compose logs backend
```

## 📞 Support

For development support:
- **Documentation**: `/api/docs`
- **Health Status**: `/health`
- **Metrics**: `/metrics`
- **Logs**: Check structured JSON logs

## 🔒 Security

This backend implements enterprise-grade security:
- **Authentication**: bcrypt + JWT + MFA
- **Authorization**: Role-based access control
- **Input Validation**: Server-side validation
- **Security Headers**: HSTS, CSP, XSS protection
- **Rate Limiting**: Per-user and per-IP limits
- **Audit Logging**: Complete security audit trail

## 📄 License

This project implements production-grade standards and security practices for enterprise deployment.