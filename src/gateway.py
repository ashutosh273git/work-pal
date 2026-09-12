# import json
# from langchain_openai import ChatOpenAI
# from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
# from src import config
# from src.logger import get_logger

# logger = get_logger(__name__)

# PRIMARY_TARGET = {"provider": "@hr-assistant",
#         "override_params": {"model": config.LLM_MODEL_NAME}}

# FALLBACK_TARGET = {"provider": "@hr-assistant-backup",
#         "override_params": {"model": "openai/gpt-oss-20b"}}

# # gateway features include loadbalancing, model routing, fallback and caching

# GATEWAY_CONFIG = {
#     "strategy":{
#         "mode": "fallback"
#     },
#     "targets": [PRIMARY_TARGET, FALLBACK_TARGET]
# }

# def get_gateway_llm() -> ChatOpenAI:
#     """Return a chat model routed through Portkey (no config/fallback - see module docstring)."""
#     logger.info("Routing LLM calls through Portkey")
#     headers = createHeaders(api_key = config.PORTKEY_API_KEY, config=GATEWAY_CONFIG)
#     return ChatOpenAI(
#         api_key=config.PORTKEY_API_KEY,
#         base_url=PORTKEY_GATEWAY_URL,
#         model=config.LLM_MODEL_NAME,
#         default_headers=headers
#     )

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from src import config
from src.logger import get_logger


logger = get_logger(__name__)

# The one Groq integration set up in the Portkey dashboard for this
# workspace (see module docstring for why there's only one).
PRIMARY_PROVIDER = "@hr-assistant"



def get_gateway_llm() -> ChatOpenAI:
    """Return a chat model routed through Portkey (no config/fallback - see module docstring)."""
    logger.info("Routing LLM calls through Portkey (provider=%s)", PRIMARY_PROVIDER)
    headers = createHeaders(api_key=config.PORTKEY_API_KEY, provider=PRIMARY_PROVIDER)
    return ChatOpenAI(
        api_key="portkey",  # dummy value - the real auth is in the headers
        base_url=PORTKEY_GATEWAY_URL,
        model=config.LLM_MODEL_NAME,
        default_headers=headers,
    )

