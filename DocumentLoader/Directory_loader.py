from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,PyPDFLoader,DirectoryLoader

loader=DirectoryLoader(path='books',glob='*.pdf',loader_cls=PyPDFLoader)

docs=loader.lazy_load()                # if we have many documents so we can not load it to memory all at one tie therefore we use concept of lazy load

for doc in docs:
    print(doc.metadata)