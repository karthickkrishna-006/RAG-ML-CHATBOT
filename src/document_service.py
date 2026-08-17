import os
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


UPLOAD_FOLDER = "data/uploads"


def save_uploaded_file(file):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(file_path)

    return file_path


def process_pdf(file_path):
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks