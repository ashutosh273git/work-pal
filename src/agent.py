from langchain.agents import create_agent
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def create_hr_agent(llm, tools):
    """Return a LangChain agent that can
    call our tools to answer questions."""
    logger.info("Creating HR agent with %d tool(s)", len(tools))
    agent = create_agent(model=llm,
                        tools=tools,
                        system_prompt=config.SYSTEM_PROMPT)
    logger.info("HR agent is ready")
    return agent