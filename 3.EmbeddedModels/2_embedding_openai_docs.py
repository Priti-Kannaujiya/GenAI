from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)
documents=[
    "Delhi is the capital of India",
    "Tajmahal is in Agra",
    "Bhopal is in MP"
]
result=embedding.embed_documents(documents)

print(result)
