from langchain_community.document_loaders import TextLoader

data=TextLoader("document loader/notes.txt")
docs = data.load()
print(len(docs))
