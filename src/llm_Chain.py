from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
import time
# To-Do, print the cost
# print 
os.environ["OPENAI_API_KEY"] = ""
    
def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0",
        verbose=True
        #default is stuffdocumentchain
    )
    prompt = PromptTemplate(
        input_variables=["nutzerfrage", "docs"],
        template="""HAUPTAUFGABE
        Du bist ein Chatbot für die Debeka Versicherungsgruppe. Du wurdest so designet das dein Fokus darauf liegt Nutzern Fragen zu Versicherungsbedingungen zu beantworten.
        VERHALTEN UND METHODEN
        Du wirst nur in Deutsch antworten. Du erhälst Textausschnitte von Dokumenten  mit wichtigem fachlichem Inhalt und eine Frage als Eingabe. Deine Aufgabe ist es die Frage nur mit Informationen aus den erhaltenen Dokumenten zu beantworten. Wenn ein Dokument nicht die notwendigen Informationen enthält um die Frage zu beantworten dann schreib: "Leider habe ich ungenügend Informationen um diese Frage zu beantworten". 
        Frage: "{nutzerfrage}"
        Textausschnitte von relevanten Dokumenten:"{docs}"
        """
    )
    return LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True
    )

if __name__ == "__main__":

    embedding = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="Debeka-Versicherungsumfang-Hausratversicherung",
        embedding_function=embedding,
        persist_directory="data/chroma",
    )
    chain = make_chain()
    
    # Die Frage wird durch die Chathistory umformuliert
    # Alles wird mit in den Prompt gegeben
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})

    while True:
        print()
        nutzerfrage = input("Frage: ")
        docs = retriever.get_relevant_documents(nutzerfrage) 
        
                
        with get_openai_callback() as cb:
            # Problem ist ich will die sachen in den context integrieren aber von hier werden die 3 Objekte einzeln in den gesamt propmt gegeben!
            response = chain.run({"nutzerfrage" : nutzerfrage, "docs" : docs})
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print(f"Calc Total Cost (USD): ${(cb.prompt_tokens/1000)*0.003 + (cb.completion_tokens/1000)*0.004}")
        # Retrieve answer
        print(f"Antwort: {response}")
        
     
