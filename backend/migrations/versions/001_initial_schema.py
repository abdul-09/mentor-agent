"""
Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-01-18 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(60), nullable=False),
        sa.Column('full_name', sa.String(100), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_admin', sa.Boolean(), nullable=False, default=False),
        sa.Column('mfa_enabled', sa.Boolean(), nullable=False, default=False),
        sa.Column('mfa_secret', sa.String(32), nullable=True),
        sa.Column('mfa_backup_codes', postgresql.JSONB(), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False, default=0),
        sa.Column('account_locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_password_change', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    
    # Create indexes for users table
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_is_active', 'users', ['is_active'])
    op.create_index('ix_users_is_verified', 'users', ['is_verified'])
    op.create_index('ix_users_email_active', 'users', ['email', 'is_active'])
    op.create_index('ix_users_created_verified', 'users', ['created_at', 'is_verified'])
    
    # Create user_sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('access_token_jti', sa.String(36), nullable=False, unique=True),
        sa.Column('refresh_token_jti', sa.String(36), nullable=False, unique=True),
        sa.Column('access_token_expires', sa.DateTime(timezone=True), nullable=False),
        sa.Column('refresh_token_expires', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='active'),
        sa.Column('device_info', postgresql.JSONB(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('location', postgresql.JSONB(), nullable=True),
        sa.Column('is_trusted_device', sa.Boolean(), nullable=False, default=False),
        sa.Column('mfa_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('security_level', sa.String(20), nullable=False, default='standard'),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('activity_count', sa.Integer(), nullable=False, default=0),
        sa.Column('revoked_reason', sa.String(100), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    
    # Create foreign key for user_sessions
    op.create_foreign_key(
        'fk_user_sessions_user_id_users',
        'user_sessions', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Create indexes for user_sessions table
    op.create_index('ix_user_sessions_user_id', 'user_sessions', ['user_id'])
    op.create_index('ix_user_sessions_access_token_jti', 'user_sessions', ['access_token_jti'])
    op.create_index('ix_user_sessions_refresh_token_jti', 'user_sessions', ['refresh_token_jti'])
    op.create_index('ix_user_sessions_status', 'user_sessions', ['status'])
    op.create_index('ix_sessions_user_status', 'user_sessions', ['user_id', 'status'])
    op.create_index('ix_sessions_user_created', 'user_sessions', ['user_id', 'created_at'])
    op.create_index('ix_sessions_access_expires', 'user_sessions', ['access_token_expires'])
    op.create_index('ix_sessions_refresh_expires', 'user_sessions', ['refresh_token_expires'])
    op.create_index('ix_sessions_last_activity', 'user_sessions', ['last_activity_at'])
    
    # Create files table
    op.create_table(
        'files',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('original_filename', sa.String(255), nullable=False),
        sa.Column('stored_filename', sa.String(255), nullable=False, unique=True),
        sa.Column('file_type', sa.String(20), nullable=False),
        sa.Column('mime_type', sa.String(100), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('file_hash', sa.String(64), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='uploading'),
        sa.Column('security_scan_status', sa.String(20), nullable=False, default='pending'),
        sa.Column('security_scan_details', postgresql.JSONB(), nullable=True),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('text_extraction_status', sa.String(20), nullable=False, default='pending'),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('embedding_status', sa.String(20), nullable=False, default='pending'),
        sa.Column('chunk_count', sa.Integer(), nullable=True),
        sa.Column('processing_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('storage_backend', sa.String(20), nullable=False, default='local'),
        sa.Column('is_public', sa.Boolean(), nullable=False, default=False),
        sa.Column('access_count', sa.Integer(), nullable=False, default=0),
        sa.Column('retention_policy', sa.String(20), nullable=False, default='standard'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create foreign key for files
    op.create_foreign_key(
        'fk_files_user_id_users',
        'files', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Create indexes for files table
    op.create_index('ix_files_user_id', 'files', ['user_id'])
    op.create_index('ix_files_file_type', 'files', ['file_type'])
    op.create_index('ix_files_status', 'files', ['status'])
    op.create_index('ix_files_file_hash', 'files', ['file_hash'])
    op.create_index('ix_files_user_status', 'files', ['user_id', 'status'])
    op.create_index('ix_files_user_created', 'files', ['user_id', 'created_at'])
    op.create_index('ix_files_type_status', 'files', ['file_type', 'status'])
    op.create_index('ix_files_expires', 'files', ['expires_at'])
    
    # Create analyses table
    op.create_table(
        'analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('file_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_name', sa.String(200), nullable=False),
        sa.Column('analysis_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('configuration', postgresql.JSONB(), nullable=True),
        sa.Column('repository_url', sa.String(500), nullable=True),
        sa.Column('repository_branch', sa.String(100), nullable=True),
        sa.Column('repository_commit', sa.String(40), nullable=True),
        sa.Column('results', postgresql.JSONB(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('api_calls_made', sa.Integer(), nullable=False, default=0),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('complexity_score', sa.Float(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('security_score', sa.Float(), nullable=True),
        sa.Column('performance_score', sa.Float(), nullable=True),
        sa.Column('maintainability_score', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('progress_percentage', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    
    # Create foreign keys for analyses
    op.create_foreign_key(
        'fk_analyses_user_id_users',
        'analyses', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_analyses_file_id_files',
        'analyses', 'files',
        ['file_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Create indexes for analyses table
    op.create_index('ix_analyses_user_id', 'analyses', ['user_id'])
    op.create_index('ix_analyses_file_id', 'analyses', ['file_id'])
    op.create_index('ix_analyses_analysis_type', 'analyses', ['analysis_type'])
    op.create_index('ix_analyses_status', 'analyses', ['status'])
    op.create_index('ix_analyses_user_type', 'analyses', ['user_id', 'analysis_type'])
    op.create_index('ix_analyses_user_created', 'analyses', ['user_id', 'created_at'])
    op.create_index('ix_analyses_status_created', 'analyses', ['status', 'created_at'])
    op.create_index('ix_analyses_file_status', 'analyses', ['file_id', 'status'])
    
    # Create qa_interactions table
    op.create_table(
        'qa_interactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=True),
        sa.Column('context_chunks', postgresql.JSONB(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('model_used', sa.String(50), nullable=True),
        sa.Column('feedback_rating', sa.Integer(), nullable=True),
        sa.Column('feedback_comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('answered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    
    # Create foreign keys for qa_interactions
    op.create_foreign_key(
        'fk_qa_interactions_analysis_id_analyses',
        'qa_interactions', 'analyses',
        ['analysis_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_qa_interactions_user_id_users',
        'qa_interactions', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Create indexes for qa_interactions table
    op.create_index('ix_qa_interactions_analysis_id', 'qa_interactions', ['analysis_id'])
    op.create_index('ix_qa_interactions_user_id', 'qa_interactions', ['user_id'])
    op.create_index('ix_qa_analysis_created', 'qa_interactions', ['analysis_id', 'created_at'])
    op.create_index('ix_qa_user_created', 'qa_interactions', ['user_id', 'created_at'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('qa_interactions')
    op.drop_table('analyses')
    op.drop_table('files')
    op.drop_table('user_sessions')
    op.drop_table('users')