import re
import os
import sys
import glob
from typing import Callable, List, Tuple, Dict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pdfminer.high_level import extract_pages, extract_text
import re

load_dotenv()
def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    
    persist_directory="src/data/chroma"
    file_path = "src/data/BHR4.pdf"
    text =extract_text(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2800, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    
    #print(len(chunks))
    #print(chunks)
    
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        collection = db.get()
        
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(chunks)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(chunks, embeddings, collection_name="Debeka_Allgemeine_Hausratversicherungsbedingungen", persist_directory=persist_directory)
    db.persist()
    db = None

    print(f"Ingestion complete!")

if __name__ == "__main__":
    main()
