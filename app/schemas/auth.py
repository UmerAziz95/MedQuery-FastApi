from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    business_client_id: str = Field(..., example="acme")
    email: EmailStr = Field(..., example="admin@acme.com")
    password: str = Field(..., example="secret")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CreateAdminRequest(BaseModel):
    business_client_id: str = Field(..., example="acme")
    email: EmailStr = Field(..., example="admin@acme.com")
    password: str = Field(..., example="secret")
    role: str = Field("admin", example="admin")
