import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data"

def load_and_split_documents() -> list[Document]:
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    documents = []
    for f in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, f)
        loader = None
        if f.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif f.endswith(".txt"):
            loader = TextLoader(file_path)
        elif f.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif f.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif f.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(file_path)

        if loader:
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def split_text_into_docs(text: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc = Document(page_content=text, metadata={"source": "text_input"})
    return text_splitter.split_documents([doc])