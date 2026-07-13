from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("document loader/CSS Notes.pdf")
docs = data.load()
print(docs[71])