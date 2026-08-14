from src.helper import *
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS


load_dotenv()


file = file_loader("Data/")
chunk_data = chunking_data(file)
embeddings = get_embedding()

docs = FAISS.from_documents(
    documents=chunk_data,
    embedding=embeddings
)