from dotenv import load_dotenv 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstores=Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriver=vectorstores.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatMistralAI(model="mistral-small-2506")

#prompt template 
prompt =ChatPromptTemplate.from_messages(
    [
        ("system",
            """you are a helpful AI assistant. 
            use only the provided context to annswer the question. 
            If the answer is not presnt in the context, 
            say: "I could not find the answer in the documnet. 
            """
         ),
         (
             "human",
             """context:
             {context}
             Question:
             {question}"""
         )
    ]
)

print("Rag system is created")

print("press 0 to exit")

while True:
    query=input("you: ")
    if query=="0":
        break
    docs=retriver.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    response=llm.invoke(final_prompt)
    print(f"\n AI : {response.content}")