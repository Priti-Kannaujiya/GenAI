from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Return sentiment of the review as positive, negative, or neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list",
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list",
    )


structured_model = model.with_structured_output(Review, method="json_mode")

result = structured_model.invoke(
    """
Return the answer as JSON.

Review:
I recently bought the boAt Airdopes 141 and the sound quality is really impressive for the price.
The battery backup lasts long, and the earbuds are comfortable for daily use. Connectivity is smooth
with almost no lag during calls or music. Overall, it's a great budget-friendly option for students
and casual users.

Pros: Good bass quality, long battery life, affordable price.
Cons: Mic quality could be better, and the case feels slightly bulky.
"""
)

print(result)