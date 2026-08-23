from langchain_community.document_loaders import TextLoader
from src import config

def load_documents(file_path: str = config.DATA_FILE_PATH):
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()