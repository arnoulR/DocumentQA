from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.schema.messages import SystemMessage
import os
import time

def make_agent():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
    )
    embedding = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="Debeka_Allgemeine_Hausratversicherungsbedingungen",
        embedding_function=embedding,
        persist_directory="data/chroma_kleinere_chunks",
    )
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})

    tool = create_retriever_tool(
        retriever, 
        "Debeka_Allgemeine_Hausratversicherungsbedingungen",
        "Sucht Dokumente zu dem Bedingungswerk der Debeka Hausratversicherung und gibt diese zurück."
    )
    system_message = SystemMessage(
        # Prompt anpassen damit Seiten richtig retrieved werden und auch sonst alles ein wenig besser klappt!
        #Manchmal sucht der Agent falsche Sachen, durch Beispiele erklären wie er suchen soll
        # Im Prompt das Dokument am besten erklären, also wie die Pakete aufgebaut sind etc.
        # Seiten angabe klappt noch nicht gut oft werden aus den mittlereen Chunks die Seitenanzahl "Übersehen" dafür wird oft noch eine Seitenzahl hinzugefügt welche garnicht vorhanden ist
        content=(
            "Du bist ein Chatbot für die Debeka Versicherungsgruppe. Deine Hauptaufgabe ist Nutzern Fragen zu Versicherungsbedingungen der Debeka zu beantworten. "
            "Antworte nur in Deutsch, beantworte nur Fragen zu den Versicherungsbedingungen der Debeka und keiner anderen Versicherung."
            "Benutze Tools falls Fragen zu Versicherungsbedingungen gestellt werden "
            "Bitte schreib bei jeder Antwort (falls vorhanden) dazu welches Dokument zum beantworten der Frage genutzt wurden und von welchen Seiten des Dokuments diese stammen. Chunks sind keine Seiten und sollten deshalb auch nicht als Seite angegeben werden! Eine Seite ist durch den Text: Seite 35 von 36, gekennzeichnet dabei ist der Text danach immer der Text der gennanten Seitenzahl, in diesem Beispiel also 35."
        )
)
    tools = [tool]
    return create_conversational_retrieval_agent(llm, 
                                                 tools, 
                                                 system_message=system_message, 
                                                 verbose=True, 
                                                 #return_intermediate_steps=True
                                                 )

if __name__ == "__main__":
    load_dotenv()
    conversational_retrieval_agent = make_agent()

    while True:
        nutzerfrage = input("Frage: ")      
                
        with get_openai_callback() as cb:
            antwort = conversational_retrieval_agent(nutzerfrage)
            
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print(f"Calc Total Cost (USD): ${(cb.prompt_tokens/1000)*0.003 + (cb.completion_tokens/1000)*0.004}\n")
        # Retrieve answer
        #print(f"Antwort: {response}\n")
        ergebnis = antwort["output"]
        print(f"{ergebnis}\n")