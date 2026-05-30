from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",task="text-generation")

model=ChatHuggingFace(llm=llm)


loader=PyPDFLoader(r'C:\Users\Priti\Desktop\Langchain_models\DocumentLoader\dl-curriculum.pdf')

docs=loader.load()

text = "\n\n".join(doc.page_content for doc in docs[:3])   #means: take the text from the first 3 pages of the PDF and combine it into one big string.

prompt = PromptTemplate(
    template="""
Answer the question using the PDF text below.

PDF text:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "context": text,
    "question": "What is this PDF about?"
})

print(result)
