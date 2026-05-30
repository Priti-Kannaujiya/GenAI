from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",task="text-generation")

model=ChatHuggingFace(llm=llm)


loader = TextLoader(
    file_path=r"C:\Users\Priti\Desktop\Langchain_models\DocumentLoader\Cricket.txt",
    encoding="utf-8"
)

prompt=PromptTemplate(
    template="Write summary of given text. \n {poem}",
    input_variables=['poem']
)

parser=StrOutputParser()
docs = loader.load()
# print(type(docs))
# print(docs[0].page_content)

chain=prompt| model | parser

print(chain.invoke({'poem':docs[0].page_content}))
