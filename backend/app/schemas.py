from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List
from uuid import UUID


# User Schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


# Prediction Schemas
class PredictionRequest(BaseModel):
    hand_landmarks: List[List[float]] = Field(..., description="Array of hand landmark coordinates")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


# Course Progress Schemas
class CourseProgressResponse(BaseModel):
    Ka: int = 0
    Kha: int = 0
    Ga: int = 0
    Gha: int = 0
    Nga: int = 0
    Cha: int = 0
    Chha: int = 0
    Ja: int = 0
    Jha: int = 0
    Yan: int = 0
    Ta: int = 0
    Tha: int = 0
    Da: int = 0
    Dha: int = 0
    Na: int = 0
    Taa: int = 0
    Thaa: int = 0
    Daa: int = 0
    Dhaa: int = 0
    Naa: int = 0
    Pa: int = 0
    Pha: int = 0
    Ba: int = 0
    Bha: int = 0
    Ma: int = 0
    Ya: int = 0
    Ra: int = 0
    La: int = 0
    Wa: int = 0
    T_Sha: int = 0
    M_Sha: int = 0
    D_Sha: int = 0
    Ha: int = 0
    Ksha: int = 0
    Tra: int = 0
    Gya: int = 0

    model_config = ConfigDict(from_attributes=True)


class IncrementCourseRequest(BaseModel):
    course_name: str = Field(..., description="Name of the course/character to increment")


class IncrementCourseResponse(BaseModel):
    message: str
    new_count: int
