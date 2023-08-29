from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

os.environ["OPENAI_API_KEY"] = ""

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0",
        # verbose=True
    )
    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="Debeka-Versicherungsumfang-Hausratversicherung",
        embedding_function=embedding,
        persist_directory="src/data/chroma",
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3}),
        return_source_documents=True,
        # verbose=True,
    )


if __name__ == "__main__":
   
    chain = make_chain()
    chat_history = []

    while True:
        print()
        question = input("Question: ")

        # Generate answer
        response = chain({"context": open_file("src/system_profile.txt"), "question": question, "chat_history": chat_history})

        # Retrieve answer
        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # Display answer
        print("\n\nSources:\n")
        for document in source:
            print(f"Page: {document.metadata['page_number']}")
            print(f"Text chunk: {document.page_content[:160]}...\n")
        print(f"Answer: {answer}")
        print(f"THIS IS THE HISTORY: {chat_history}")
