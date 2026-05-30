from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",task="text-generation")
model=ChatHuggingFace(llm=llm)

# schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""I recently bought the boAt Airdopes 141 and the 
             sound quality is really impressive for the price. 
             The battery backup lasts long, and the earbuds are 
             comfortable for daily use. Connectivity is smooth with almost no lag during calls or music. 
             Overall, it’s a great budget-friendly option for students and casual users.
             Pros: Good bass quality, long battery life, affordable price.
             Cons: Mic quality could be better, and the case feels slightly bulky.

Review by Nitish Singh
""")

print(result)