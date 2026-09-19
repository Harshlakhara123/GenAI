from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10,default=5,description="Cumulative Grade Point Average of the student, should be between 0 and 10")

newStudent = {'name': 'John Doe'}

student = Student(**newStudent)

studentDict = dict(student)
studentJson = student.model_dump_json()