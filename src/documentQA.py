from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
import time
# To-Do, print the cost
# print 
os.environ["OPENAI_API_KEY"] = ""
context = """HAUPTAUFGABE
Du bist ein Chatbot für die Debeka Versicherungsgruppe. Du wurdest so designet das dein Fokus darauf liegt Nutzern Fragen zu Versicherungsbedingungen zu beantworten.

VERHALTEN UND METHODEN
Du wirst nur in Deutsch antworten. Du erhälst Textausschnitte von Dokumenten  mit wichtigem fachlichem Inhalt und eine Frage als Eingabe. Deine Aufgabe ist es die Frage nur mit Informationen aus den erhaltenen Dokumenten zu beantworten.
Wenn ein Dokument nicht die notwendigen Informationen enthält um die Frage zu beantworten dann schreib: "Leider habe ich ungenügend Informationen um diese Frage zu beantworten. Könntest du mir mehr Details zu deiner Frage geben?".
"""
    
def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0",
        verbose=True
        #default is stuffdocumentchain
    )
    embedding = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="Debeka-Versicherungsumfang-Hausratversicherung",
        embedding_function=embedding,
        persist_directory="src/data/chroma",
    )
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

    # Die Frage wird durch die Chathistory umformuliert
    # Alles wird mit in den Prompt gegeben
    return RetrievalQA.from_chain_type(llm=model, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)



if __name__ == "__main__":
    chain = make_chain()
     

    while True:
        print()
        nutzerfrage =input("Frage: ")
        
        

        # Generate answer
        
        with get_openai_callback() as cb:
            # Problem ist ich will die sachen in den context integrieren aber von hier werden die 3 Objekte einzeln in den gesamt propmt gegeben!
            llm_response = chain(nutzerfrage)
            # process_llm_response(llm_response)
            print(llm_response)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print(f"Calc Total Cost (USD): ${(cb.prompt_tokens/1000)*0.003 + (cb.completion_tokens/1000)*0.004}")
       
        
     
