from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

def word_count(text):
    return len(text.split())

template=PromptTemplate(
    template="Write a sort poem on topic {topic}",
    input_variables=['topic']
)
parser=StrOutputParser()

chain=RunnableSequence(template,model,parser)

parallel_chain=RunnableParallel({
    'Poem':RunnablePassthrough(),
    'Word_count':RunnableLambda(word_count)
})

final_chain=chain|parallel_chain

result=final_chain.invoke({'topic':'Machine Learning'})
final_result="""{} \n word_count-{}""".format(result['Poem'],result['Word_count'])

print(final_result)