from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents=[
    'Delhi is the capital of India',
    'Azamgarh is in UP',
    'Varanasi is famous for their temples'
]

text="Delhi is the capital of India"

vector=embedding.embed_query(text)

result=embedding.embed_documents(documents)

print(str(vector))