from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person:Person={'name':'Priti','age':135}

print(new_person)