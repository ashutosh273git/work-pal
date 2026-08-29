from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def split_into_chunks(documents):
    logger.info("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)
    logger.info("Split document(s) into %d chunk(s)", len(chunks))
    return chunks