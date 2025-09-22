"""
Add password reset columns to users table

Revision ID: 002_add_password_reset_columns
Revises: 001_initial_schema
Create Date: 2025-09-21 08:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_password_reset_columns'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add password reset columns to users table."""
    
    # Add password reset columns to users table
    op.add_column('users', sa.Column('password_reset_token', sa.String(255), nullable=True))
    op.add_column('users', sa.Column('password_reset_expires', sa.DateTime(timezone=True), nullable=True))
    
    # Create index for password reset token for faster lookups
    op.create_index('ix_users_password_reset_token', 'users', ['password_reset_token'])


def downgrade() -> None:
    """Remove password reset columns from users table."""
    
    # Drop index first
    op.drop_index('ix_users_password_reset_token', table_name='users')
    
    # Drop columns
    op.drop_column('users', 'password_reset_expires')
    op.drop_column('users', 'password_reset_token')