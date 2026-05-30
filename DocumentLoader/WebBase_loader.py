import os
from langchain_community.document_loaders import WebBaseLoader

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

url = "https://www.amazon.in/dp/B0GVFJXWLQ?ref=cm_sw_r_ud_dp_X5WDFTA3GA8PN1EKWZFJ&ref_=cm_sw_r_ud_dp_X5WDFTA3GA8PN1EKWZFJ&social_share=cm_sw_r_ud_dp_X5WDFTA3GA8PN1EKWZFJ&language=en-IN"
loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))