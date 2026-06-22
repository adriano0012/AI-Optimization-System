"""
Database Models
SQLAlchemy ORM models for persistent state.
"""

import time
import uuid
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, Text,
    DateTime, JSON, ForeignKey, Index, event
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class Organization(Base):
    __tablename__ = 'organizations'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    plan = Column(String(50), default='free')
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    users = relationship('User', back_populates='organization')
    api_keys = relationship('ApiKey', back_populates='organization')
    quotas = relationship('Quota', back_populates='organization')

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'plan': self.plan,
            'created_at': str(self.created_at), 'updated_at': str(self.updated_at),
        }


class User(Base):
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255))
    role = Column(String(50), default='member')
    is_active = Column(Boolean, default=True)
    organization_id = Column(String(36), ForeignKey('organizations.id'))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    organization = relationship('Organization', back_populates='users')

    def to_dict(self):
        return {
            'id': self.id, 'email': self.email, 'name': self.name,
            'role': self.role, 'is_active': self.is_active,
            'organization_id': self.organization_id,
            'created_at': str(self.created_at),
        }


class ApiKey(Base):
    __tablename__ = 'api_keys'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    key = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    organization_id = Column(String(36), ForeignKey('organizations.id'))
    user_id = Column(String(36), ForeignKey('users.id'))
    is_active = Column(Boolean, default=True)
    rate_limit = Column(Integer, default=1000)
    created_at = Column(DateTime, server_default=func.now())
    last_used_at = Column(DateTime)

    organization = relationship('Organization', back_populates='api_keys')
    user = relationship('User')

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'key_prefix': self.key[:8] + '...',
            'is_active': self.is_active, 'rate_limit': self.rate_limit,
            'created_at': str(self.created_at),
            'last_used_at': str(self.last_used_at) if self.last_used_at else None,
        }


class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey('users.id'))
    organization_id = Column(String(36), ForeignKey('organizations.id'))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(36))
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_org', 'organization_id'),
        Index('idx_audit_created', 'created_at'),
    )

    def to_dict(self):
        return {
            'id': self.id, 'user_id': self.user_id,
            'organization_id': self.organization_id,
            'action': self.action, 'resource_type': self.resource_type,
            'resource_id': self.resource_id, 'details': self.details,
            'ip_address': self.ip_address, 'created_at': str(self.created_at),
        }


class OptimizationRecord(Base):
    __tablename__ = 'optimization_records'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey('users.id'))
    organization_id = Column(String(36), ForeignKey('organizations.id'))
    prompt_hash = Column(String(64))
    original_length = Column(Integer)
    optimized_length = Column(Integer)
    compression_ratio = Column(Float)
    task_type = Column(String(100))
    model_used = Column(String(100))
    latency_ms = Column(Float)
    tokens_saved = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    extra_data = Column("metadata", JSON)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index('idx_opt_user', 'user_id'),
        Index('idx_opt_org', 'organization_id'),
        Index('idx_opt_created', 'created_at'),
        Index('idx_opt_task', 'task_type'),
    )

    def to_dict(self):
        return {
            'id': self.id, 'task_type': self.task_type,
            'compression_ratio': self.compression_ratio,
            'latency_ms': self.latency_ms, 'tokens_saved': self.tokens_saved,
            'success': self.success, 'created_at': str(self.created_at),
            'metadata': self.extra_data,
        }


class Quota(Base):
    __tablename__ = 'quotas'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    organization_id = Column(String(36), ForeignKey('organizations.id'))
    quota_type = Column(String(50), nullable=False)
    limit_value = Column(Integer, nullable=False)
    used_value = Column(Integer, default=0)
    period = Column(String(20), default='monthly')
    reset_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    organization = relationship('Organization', back_populates='quotas')

    def to_dict(self):
        return {
            'id': self.id, 'quota_type': self.quota_type,
            'limit_value': self.limit_value, 'used_value': self.used_value,
            'period': self.period, 'reset_at': str(self.reset_at) if self.reset_at else None,
        }


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str = "sqlite:///data/uai_optimizer.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def get_session(self):
        session = self.SessionLocal()
        try:
            return session
        except Exception:
            session.close()
            raise

    def drop_tables(self):
        Base.metadata.drop_all(self.engine)
