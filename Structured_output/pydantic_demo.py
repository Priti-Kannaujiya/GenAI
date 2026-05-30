from pydantic import BaseModel,EmailStr
from typing import Optional

class Student(BaseModel):
    name:str='Priti'
    age: Optional[int]=None
    email:EmailStr

new_student={'age':'32','email':'priti@gmail.com'}     # implicit type conversion that will convert the value in the given format even if wrong format value is given "32"

student=Student(**new_student)

print(student)