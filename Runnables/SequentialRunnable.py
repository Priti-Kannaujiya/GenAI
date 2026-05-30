from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="Write a sort poem on topic {topic}",
    input_variables=['topic']
)
template2=PromptTemplate(
    template="Explain the poem {text}",
    input_variables=['text']
)
parser=StrOutputParser()

chain=RunnableSequence(template1,model,parser,template2,model,parser)
result=chain.invoke({'topic':'Environment'})
print(result)