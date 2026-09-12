from src.gateway import get_gateway_llm
from src.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """Return a Groq chat model. Reads GROQ_API_KEY from the environment"""
    logger.info("Initializing LLM via PORTKEY")
    return get_gateway_llm()