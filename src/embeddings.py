from langchain_community.embeddings import JinaEmbeddings
from src import config

def get_embedding_model():
    embeddings_model = JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return embeddings_model