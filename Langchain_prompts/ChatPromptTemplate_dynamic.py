from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",task="text-generation")
model=ChatHuggingFace(llm=llm)

# chat_template=ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {domain} assistent"),
#     HumanMessage(content="Explain in simple term what is {topic}")
# ])                             INSTEAD OF WRITING LIKE THIS TO GIVE PROMPT TO SYSTEM AND HUMAN MESSAGE  IT WILL GIVE THE INPUT AS OUTPUT

#   INSTEAD WE PASS STRING
chat_template=ChatPromptTemplate([
    ('system',"You are a helpful {domain} assistent"),
    ('human',"Explain in simple term what is {topic}")
])  

prompt=chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)