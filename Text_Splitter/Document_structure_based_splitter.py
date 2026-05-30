from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

code="""
class TextSplitter:
    
    def __init__(self, text, chunk_size, chunk_overlap):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self):
        chunks = []
        start = 0

        while start < len(self.text):
            end = start + self.chunk_size
            
            chunk = self.text[start:end]
            chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=350,
    chunk_overlap=0
)
chunks=splitter.split_text(code)
print(chunks[0])