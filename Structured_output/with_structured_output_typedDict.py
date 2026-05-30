from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Optional,Literal,Annotated     #annotation make our llm to understand what to do if it is not getting through a single word

load_dotenv()
llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",task="text-generation")
model=ChatHuggingFace(llm=llm)

class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down all the key themes discussed in the review in a list"]
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[Literal['pos','neg'],"Return sentiment of the review either negative, positive or neutral"]
    pros:Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]


structured_model=model.with_structured_output(Review)
result=structured_model.invoke("""I recently bought the boAt Airdopes 141 and the 
             sound quality is really impressive for the price. 
             The battery backup lasts long, and the earbuds are 
             comfortable for daily use. Connectivity is smooth with almost no lag during calls or music. 
             Overall, it’s a great budget-friendly option for students and casual users.
             Pros: Good bass quality, long battery life, affordable price.
             Cons: Mic quality could be better, and the case feels slightly bulky.""")

print(result)
print(result['summary'])
print(result['sentiment'])




