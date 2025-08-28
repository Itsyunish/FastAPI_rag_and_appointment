from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Annotated
from datetime import datetime

class UserInfo(BaseModel):
    name: Annotated[str, Field(max_length=50)]
    email: Annotated[EmailStr, Field(...)]
    appointment_date: str
    appointment_time: str

    @field_validator("name")
    def validate_name(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        return v

    @field_validator("appointment_date")
    def validate_date(cls, v):
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("appointment_time")
    def validate_time(cls, v):
        datetime.strptime(v, "%H:%M")
        return v
