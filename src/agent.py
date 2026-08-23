from langchain.agents import create_agent
from src import config

def create_hr_agent(llm, tools):
    """Return a LangChain agent that can
    call our tools to answer questions."""
    agent = create_agent(model=llm,
                        tools=tools,
                        system_prompt=config.SYSTEM_PROMPT)
    return agent