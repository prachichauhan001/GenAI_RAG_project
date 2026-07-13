from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader   
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data=TextLoader("document loader/notes.txt")
docs = data.load()

template=ChatPromptTemplate.from_messages([ 
    ("system","you are a ai and summraises the text"),
    ("human","{data}")
])

model=ChatMistralAI(
    model="mistral-medium-3-5"
)

prompt=template.format_messages(data=docs[0].page_content)

response=model.invoke(prompt)
print(response.content)
