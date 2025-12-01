from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import os
import shutil

DATA_PATH = "data_sources"
CHROMA_PATH = "chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    docs = loader.load()
    return docs

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100
    )
    return splitter.split_documents(documents)

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = GPT4AllEmbeddings(
        model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
        gpt4all_kwargs={"allow_download": "True"}
    )
    
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print("Vector DB Created Successfully")

def create_vector_db():
    docs = load_documents()
    chunks = split_text(docs)
    save_to_chroma(chunks)

if __name__ == "__main__":
    create_vector_db()
