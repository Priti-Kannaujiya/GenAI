from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


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

prompt1 = template1.invoke({'topic': 'blackhole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
result2 = model.invoke(prompt2)
print(result2.content)