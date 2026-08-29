from langchain_community.embeddings import JinaEmbeddings
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def get_embedding_model():
    logger.info("Initializing the embeddings model '%s'", config.EMBEDDING_MODEL_NAME)
    embeddings_model = JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    logger.info("Loaded embeddings model")
    return embeddings_model