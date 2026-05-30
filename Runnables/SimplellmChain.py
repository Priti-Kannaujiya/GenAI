from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
template=PromptTemplate(
    template='Give me 5 facts of{topic}',
    input_variables=['topic'],
)

# create an LLM chain
chain=LLMChain(llm=llm,prompt=template)

topic=input('Support vector machine')
output=chain.run(topic)

print("Generate a blog title",output)
