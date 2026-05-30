from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


sentiment_parser = PydanticOutputParser(pydantic_object=Feedback)
text_parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback as positive or negative.\n"
        "{feedback}\n"
        "{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": sentiment_parser.get_format_instructions()
    }
)

classifier_chain = template1 | model | sentiment_parser

pos_template = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

neg_template = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        pos_template | model | text_parser
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        neg_template | model | text_parser
    ),
    RunnableLambda(lambda x: "Could not find the sentiment.")
)

final_chain = RunnableParallel(
    {
        "feedback": lambda x: x["feedback"],
        "sentiment": classifier_chain
    }
) | branch_chain

result = final_chain.invoke({"feedback": "this is a terrible phone"})
print(result)

final_chain.get_graph().print_ascii()