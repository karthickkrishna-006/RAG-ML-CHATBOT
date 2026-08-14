from flask import Flask, render_template, request

from src.helper import get_embedding
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from src.prompt import *
from store import docs

import os


app = Flask(__name__)

load_dotenv()


# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# Load embeddings
embeddings = get_embedding()


# Create retriever
retriever = docs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Create Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)


# Create prompt template
base_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])