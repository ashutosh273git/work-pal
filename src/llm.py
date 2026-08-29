from langchain_groq import ChatGroq
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """Return a Groq chat model. Reads GROQ_API_KEY from the environment"""
    logger.info("Initializing LLM '%s'", config.LLM_MODEL_NAME)
    return ChatGroq(model=config.LLM_MODEL_NAME, temperature=0)