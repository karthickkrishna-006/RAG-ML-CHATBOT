import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from src.helper import get_embedding
from src.document_service import process_pdf


load_dotenv()

VECTORSTORE_PATH = "vectorstore"

embeddings = get_embedding()


# ==========================================
# CREATE VECTORSTORE
# ==========================================

def create_vectorstore(chunks):

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(
        VECTORSTORE_PATH,
        exist_ok=True
    )

    db.save_local(
        VECTORSTORE_PATH
    )

    return db


# ==========================================
# LOAD VECTORSTORE
# ==========================================

def load_vectorstore():

    index_file = os.path.join(
        VECTORSTORE_PATH,
        "index.faiss"
    )

    if not os.path.exists(index_file):
        return None

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ==========================================
# ADD DOCUMENT
# ==========================================

def add_document_to_vectorstore(file_path):

    chunks = process_pdf(file_path)

    db = load_vectorstore()

    if db is None:

        db = create_vectorstore(chunks)

    else:

        db.add_documents(chunks)

        db.save_local(
            VECTORSTORE_PATH
        )

    return db, len(chunks)