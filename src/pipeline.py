from src import config
from src.agent import create_hr_agent
from src.document_loader import load_documents
from src.llm import get_llm
from src.splitter import split_into_chunks
from src.vector_store import(
    build_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever
)
from src.tools import create_search_tool
from src.logger import get_logger
from src.tracing import check_langsmith_tracing
from src.guardrails import REFUSAL_MESSSAGE, check_input, check_output

logger =  get_logger(__name__)

def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
    """Load + split + embed the document, 
    reusing the Qdrant Cloud collection if we have one."""
    if vector_store_exists():
        print("Found an existing Qdrant Cloud collection, connecting to it (fast, no re-embedding).")
        logger.info("Qdrant Cloud collection already exists, reusing it")
        return load_vector_store()

    print("No Qdrant Cloud collection found, building one from scratch...")
    logger.info("No Qdrant Cloud collection found, building one from scratch")
    documents = load_documents(file_path)
    chunks = split_into_chunks(documents)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks")

    vector_store = build_vector_store(chunks)
    print("Vector store built and saved to disk for next time")
    return vector_store

def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the full RAG agent, ready to answer questions."""
    logger.info("Building the HR Assistant")
    config.check_api_keys()
    check_langsmith_tracing()

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)    

    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])

    logger.info("HR assistant is ready to take questions")
    return agent

def ask(agent, question: str) -> str:
    """Ask the agent a question and
    return its final answer as plain text."""
    logger.info("User question: %s", question)

    input_safe, _ = check_input(question)
    if not input_safe:
        return REFUSAL_MESSSAGE
    
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = response["messages"][-1].content
    logger.info("Final answer: %s", answer)

    output_safe, _ = check_output(answer)
    if not output_safe:
        return REFUSAL_MESSSAGE 

    return answer