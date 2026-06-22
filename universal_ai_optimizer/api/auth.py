"""
Authentication & Authorization
JWT tokens, API key validation, RBAC.
"""

import hashlib
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from passlib.context import CryptContext
    PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

JWT_SECRET = os.environ.get("UAI_JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.environ.get("UAI_JWT_EXPIRATION_HOURS", "24"))
API_KEY_LENGTH = 48


class AuthManager:
    """Manages authentication, authorization, and API keys."""

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._api_key_cache: Dict[str, Dict] = {}

    def hash_password(self, password: str) -> str:
        if PASSLIB_AVAILABLE:
            return PWD_CONTEXT.hash(password)
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, hashed: str) -> bool:
        if PASSLIB_AVAILABLE:
            return PWD_CONTEXT.verify(password, hashed)
        return hashlib.sha256(password.encode()).hexdigest() == hashed

    def create_user(self, email: str, password: str, name: str,
                    organization_id: Optional[str] = None,
                    role: str = "member") -> Dict[str, Any]:
        if not self.db:
            raise RuntimeError("Database not configured")

        session = self.db.get_session()
        try:
            from api.models import User
            existing = session.query(User).filter_by(email=email.lower()).first()
            if existing:
                raise ValueError("Email already registered")

            user = User(
                email=email.lower(),
                hashed_password=self.hash_password(password),
                name=name,
                role=role,
                organization_id=organization_id,
            )
            session.add(user)
            session.commit()
            return user.to_dict()
        finally:
            session.close()

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        if not self.db:
            return None

        session = self.db.get_session()
        try:
            from api.models import User
            user = session.query(User).filter_by(email=email.lower()).first()
            if user and self.verify_password(password, user.hashed_password):
                return user.to_dict()
            return None
        finally:
            session.close()

    def create_jwt_token(self, user_id: str,
                         organization_id: Optional[str] = None,
                         role: str = "member") -> str:
        if not JWT_AVAILABLE:
            return f"mock_token_{user_id}_{int(time.time())}"

        payload = {
            'sub': user_id,
            'org_id': organization_id,
            'role': role,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not JWT_AVAILABLE:
            return {'sub': token.split('_')[2] if '_mock_token_' in token else 'unknown'}

        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    def generate_api_key(self, name: str, organization_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         rate_limit: int = 1000) -> Dict[str, Any]:
        key = "uai_" + secrets.token_hex(API_KEY_LENGTH // 2)

        if self.db:
            session = self.db.get_session()
            try:
                from api.models import ApiKey
                api_key = ApiKey(
                    key=hashlib.sha256(key.encode()).hexdigest(),
                    name=name,
                    organization_id=organization_id,
                    user_id=user_id,
                    rate_limit=rate_limit,
                )
                session.add(api_key)
                session.commit()
                return {
                    'id': api_key.id, 'name': name, 'key': key,
                    'key_prefix': key[:12] + '...', 'is_active': True,
                    'rate_limit': rate_limit,
                    'created_at': str(api_key.created_at),
                }
            finally:
                session.close()

        return {
            'id': secrets.token_hex(8), 'name': name, 'key': key,
            'key_prefix': key[:12] + '...', 'is_active': True,
            'rate_limit': rate_limit, 'created_at': datetime.utcnow().isoformat(),
        }

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash in self._api_key_cache:
            return self._api_key_cache[key_hash]

        if self.db:
            session = self.db.get_session()
            try:
                from api.models import ApiKey
                db_key = session.query(ApiKey).filter_by(
                    key=key_hash, is_active=True
                ).first()
                if db_key:
                    info = {
                        'id': db_key.id, 'name': db_key.name,
                        'organization_id': db_key.organization_id,
                        'user_id': db_key.user_id,
                        'rate_limit': db_key.rate_limit,
                    }
                    self._api_key_cache[key_hash] = info
                    db_key.last_used_at = datetime.utcnow()
                    session.commit()
                    return info
            finally:
                session.close()
        return None

    def check_permission(self, role: str, action: str) -> bool:
        permissions = {
            'admin': ['read', 'write', 'delete', 'manage_users', 'manage_org', 'admin'],
            'member': ['read', 'write'],
            'viewer': ['read'],
        }
        return action in permissions.get(role, [])

    def revoke_api_key(self, key_id: str) -> bool:
        if not self.db:
            return False
        session = self.db.get_session()
        try:
            from api.models import ApiKey
            key = session.query(ApiKey).filter_by(id=key_id).first()
            if key:
                key.is_active = False
                session.commit()
                key_hash = key.key
                self._api_key_cache.pop(key_hash, None)
                return True
            return False
        finally:
            session.close()
