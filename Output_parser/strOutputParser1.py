from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",task="text-generation")
model=ChatHuggingFace(llm=llm)

# 1st Prompt-> Detailed report

template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd Prompt-> Summary
template2=PromptTemplate(
    template='Write a five line summary on following text.\n {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1|model|parser|template2|model|parser
result=chain.invoke({'topic':'blackhole'})
print(result)

