from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Generate a 4 line theory on the following topic {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='give a linkdin post about the topic{topic}',
    input_variables=['topic']
)
parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser)

chain_parallel=RunnableParallel({
    'theory':RunnablePassthrough(),
    'linkdin post':RunnableSequence(prompt2,model,parser)
})
final_chain=RunnableSequence(chain,chain_parallel)

result=final_chain.invoke({'topic':'AI'})
print(result)