from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser   # Json output parser does not provide schema

load_dotenv()

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",task="text-generation")
model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()

template=PromptTemplate(
    template='Give me 5 facts of{topic}.\n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
# prompt=template.format()
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)
# print(type(final_result))               Instead of writing this we can take the help of chain

chain=template|model|parser
result=chain.invoke({'topic':'Machine learning'})
print(result)


# print(result.content)
# print(prompt)