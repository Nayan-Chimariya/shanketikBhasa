from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator
import uuid
from .database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type. Uses MySQL's CHAR(36) for MySQL."""
    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif isinstance(value, uuid.UUID):
            return str(value)
        else:
            return str(uuid.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            return uuid.UUID(value)


class User(Base):
    __tablename__ = "users"

    id = Column(GUID, primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)

    # Relationship
    course_progress = relationship("UserCourseProgress", back_populates="user", uselist=False)


class UserCourseProgress(Base):
    __tablename__ = "user_course_progress"

    id = Column(GUID, primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(GUID, ForeignKey("users.id"), unique=True, nullable=False)

    # Consonants
    Ka = Column(Integer, default=0)
    Kha = Column(Integer, default=0)
    Ga = Column(Integer, default=0)
    Gha = Column(Integer, default=0)
    Nga = Column(Integer, default=0)
    Cha = Column(Integer, default=0)
    Chha = Column(Integer, default=0)
    Ja = Column(Integer, default=0)
    Jha = Column(Integer, default=0)
    Yan = Column(Integer, default=0)
    Ta = Column(Integer, default=0)
    Tha = Column(Integer, default=0)
    Da = Column(Integer, default=0)
    Dha = Column(Integer, default=0)
    Na = Column(Integer, default=0)
    Taa = Column(Integer, default=0)
    Thaa = Column(Integer, default=0)
    Daa = Column(Integer, default=0)
    Dhaa = Column(Integer, default=0)
    Naa = Column(Integer, default=0)
    Pa = Column(Integer, default=0)
    Pha = Column(Integer, default=0)
    Ba = Column(Integer, default=0)
    Bha = Column(Integer, default=0)
    Ma = Column(Integer, default=0)
    Ya = Column(Integer, default=0)
    Ra = Column(Integer, default=0)
    La = Column(Integer, default=0)
    Wa = Column(Integer, default=0)

    # Sibilants & Others
    T_Sha = Column(Integer, default=0)
    M_Sha = Column(Integer, default=0)
    D_Sha = Column(Integer, default=0)
    Ha = Column(Integer, default=0)
    Ksha = Column(Integer, default=0)
    Tra = Column(Integer, default=0)
    Gya = Column(Integer, default=0)

    # Relationship
    user = relationship("User", back_populates="course_progress")
