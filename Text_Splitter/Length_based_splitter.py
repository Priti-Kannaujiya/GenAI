from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


loader = TextLoader(
    file_path=r"C:\Users\Priti\Desktop\Langchain_models\DocumentLoader\Cricket.txt",
    encoding="utf-8"
)
docs = loader.load()

# text="""Machine Learning is a branch of Artificial Intelligence.
# It enables computers to learn from data.
# ML systems improve automatically through experience.
# Machine Learning is widely used in modern technology.
# Recommendation systems use Machine Learning algorithms.
# Netflix and YouTube suggest videos using ML.
# Online shopping apps also use ML for recommendations.
# Spam email detection is another application of ML.
# ML helps in image and speech recognition.
# Self-driving cars depend heavily on Machine Learning.
# Healthcare industries use ML for disease prediction.
# Banks use ML for fraud detection systems.
# Machine Learning requires large amounts of data.
# Data is used to train ML models.
# Training helps models identify patterns.
# There are three main types of Machine Learning.

# """

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=' '
)
result=splitter.split_documents(docs)

print(result)