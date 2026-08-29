from langchain_community.document_loaders import TextLoader
from src import config
from src.logger import  get_logger

logger = get_logger(__name__)

def load_documents(file_path: str = config.DATA_FILE_PATH):
    logger.info("Loading documents from document loader", file_path)
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    logger.info("Loaded %d documents", len(documents))
    return documents