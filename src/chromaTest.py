from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
import time

if __name__ == "__main__":
    load_dotenv()
    
    embedding = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="Debeka-Versicherungsumfang-Hausratversicherung",
        embedding_function=embedding,
        persist_directory="data/chroma",
        # Hier steckt der Fehler!, von src zu starten bedeutet src muss aus dem Pfad...
    )
    
    query = "Wie sind Brandschäden abgesichert?"
    docs = vector_store.similarity_search(query)
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
    docs2 = retriever.get_relevant_documents(query)
    print(len(docs))
    #print(docs[0])
    ("DOCS 2 AB HIER")
    print(len(docs2))
    print(docs2[1])
    #while True:
        #nutzerfrage = input("Frage: ")
        

        
        
        

    
        
     
