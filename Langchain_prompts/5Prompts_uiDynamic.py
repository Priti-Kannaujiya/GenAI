from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input=st.selectbox("Select research paper name",["Attention is all you need","BERT:Pre-training of Deep Bidirectional Transformers",
                                       "GPT3:Language models are few shot learner","Diffusion model beat GANS on Image Synthesis"])

style_input=st.selectbox("select explanation style",["beginner-friendly","Technical","Code-oriented","Mathematical"])

length_input=st.selectbox("Select explanation length",["short(10-11 paragraph)","Medium(15-20 paragraph)","long(20-30 paragraph)"])


# TEMPLATE

# template=PromptTemplate(
#     template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style: {style_input}
# Explanation Length: {length_input}

# 1. Mathematical Details:
#    - Include relevant mathematical equations if present in the paper.
#    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

# 2. Analogies:
#    - Use relatable analogies to simplify complex ideas.

# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

# Ensure the summary is clear, accurate, and aligned with the provided style and length.
# """,
# input_variables=['paper_input','style_input','length_input']
# )


# FILL THE PALCEHOLDER
template=load_prompt('template.json')

prompt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
,
})

if st.button('summarize'):
    result=model.invoke(prompt)
    st.write(result.content)