import os
from langchain_community.vectorstores import FAISS
from src import config
from src.embeddings import get_embedding_model
from src.logger import get_logger

logger = get_logger(__name__)

# building a vector store
def build_vector_store(chunks):
    """Embed every chunk and build a searchable FAISS index in memory"""
    logger.info("Embeddingg %d chunks and building FAISS index")
    embeddings_model = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    logger.info("FAISS index built in memory")
    return vector_store

# save vector store
def save_vector_store(vector_store, path:str = config.VECTOR_STORE_PATH) -> None:
    """Save the FAISS index to disk
    so we dont have to rebuild it every time"""
    vector_store.save_local(path)
    logger.info("Saved FAISS index to '%s'", path)

# load vector store
def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """Load a previously saved FAISS index from disk"""
    logger.info("Loading FAISS index from '%s'", path)
    embedding_model = get_embedding_model()
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
    """Check if a saved FAISS index already exists on disk"""
    return os.path.exists(os.path.join(path, "index.faiss"))

def get_retriever(vector_store, k:int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever that 
    returns the top k matching chunks"""
    logger.info("Creating a retriever with top_k=%d", k)
    return vector_store.as_retriever(search_kwargs={"k": k})