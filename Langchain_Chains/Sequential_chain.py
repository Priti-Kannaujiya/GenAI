from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate interesting facts about {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Find top five facts from this text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt | model | parser | template2 | model | parser

result = chain.invoke({"topic": "cricket"})
print(result)

chain.get_graph().print_ascii()