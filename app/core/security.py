from datetime import datetime, timedelta
from typing import Any

from jose import jwt
from passlib.context import CryptContext

from app.core.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload:
    def __init__(self, sub: str, business_id: str | None, role: str):
        self.sub = sub
        self.business_id = business_id
        self.role = role


def create_access_token(subject: str, business_id: str | None, role: str) -> str:
    settings = get_settings()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    to_encode: dict[str, Any] = {
        "sub": subject,
        "business_id": business_id,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def decode_token(token: str) -> TokenPayload:
    settings = get_settings()
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    return TokenPayload(
        sub=payload.get("sub"),
        business_id=payload.get("business_id"),
        role=payload.get("role"),
    )
