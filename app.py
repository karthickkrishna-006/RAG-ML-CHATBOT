from flask import Flask, render_template, request, jsonify

from src.document_service import save_uploaded_file
from src.prompts import system_prompt

from store import load_vectorstore, add_document_to_vectorstore

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

import os
import re


# ==========================================
# APP CONFIGURATION
# ==========================================

app = Flask(__name__)

load_dotenv()


# ==========================================
# GROQ API KEY
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")


# ==========================================
# LOAD VECTORSTORE
# ==========================================

docs = load_vectorstore()

if docs is not None:

    retriever = docs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

else:

    retriever = None


# ==========================================
# GROQ LLM
# ==========================================

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.4,
    api_key=GROQ_API_KEY
)


# ==========================================
# PROMPT
# ==========================================

base_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# ==========================================
# DOCUMENT CHAIN
# ==========================================

question_answer_chain = create_stuff_documents_chain(
    llm,
    base_prompt
)


# ==========================================
# RAG CHAIN
# ==========================================

rag_chain = None

if retriever is not None:

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )


# ==========================================
# REMOVE MODEL THINKING
# ==========================================

def clean_answer(answer):

    if not answer:
        return ""

    answer = str(answer)

    # Remove <think>...</think>
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove an unmatched <think>
    answer = re.sub(
        r"<think>.*$",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove unmatched </think>
    answer = re.sub(
        r"</think>",
        "",
        answer,
        flags=re.IGNORECASE
    )

    return answer.strip()


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def index():

    return render_template("chat.html")


# ==========================================
# CHAT API
# ==========================================

@app.route("/get", methods=["GET", "POST"])
def chat():

    global rag_chain

    msg = request.form.get("msg")

    if not msg:

        return jsonify({
            "success": False,
            "error": "Message is required"
        }), 400


    # No documents available
    if rag_chain is None:

        return jsonify({
            "success": False,
            "error": "Please upload a document before asking questions."
        }), 400


    try:

        response = rag_chain.invoke({
            "input": msg
        })

        answer = response.get("answer", "")

        # IMPORTANT:
        # Show only the final answer.
        # Remove model reasoning / <think> blocks.
        answer = clean_answer(answer)

        print("Question:", msg)
        print("Final Answer:", answer)

        return answer


    except Exception as e:

        print("Chat error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==========================================
# DOCUMENT UPLOAD
# ==========================================

@app.route("/api/documents/upload", methods=["POST"])
def upload_document():

    global docs
    global retriever
    global rag_chain


    # --------------------------------------
    # Check file
    # --------------------------------------

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "error": "No file uploaded"
        }), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400


    # --------------------------------------
    # Validate PDF
    # --------------------------------------

    if not file.filename.lower().endswith(".pdf"):

        return jsonify({
            "success": False,
            "error": "Only PDF files are currently supported"
        }), 400


    try:

        # ----------------------------------
        # Save uploaded file
        # ----------------------------------

        file_path = save_uploaded_file(file)


        # ----------------------------------
        # Add document to FAISS
        # ----------------------------------

        docs, chunk_count = add_document_to_vectorstore(
            file_path
        )


        # ----------------------------------
        # Refresh retriever
        # ----------------------------------

        retriever = docs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )


        # ----------------------------------
        # Rebuild RAG chain
        # ----------------------------------

        rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )


        print(
            f"Uploaded: {file.filename} | "
            f"Chunks: {chunk_count}"
        )


        return jsonify({

            "success": True,

            "filename": file.filename,

            "chunks": chunk_count,

            "message": (
                "Document uploaded and indexed successfully"
            )
        })


    except Exception as e:

        print("Upload error:", e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8080,
        debug=False
    )