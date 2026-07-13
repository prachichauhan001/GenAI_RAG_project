from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
    )


data=PyPDFLoader("document loader/CSS Notes.pdf")
docs = data.load()
chunks=text_splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)