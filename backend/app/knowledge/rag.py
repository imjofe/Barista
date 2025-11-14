"""RAG chain for menu question answering."""

from __future__ import annotations

from typing import List

from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import Chroma

from app.llm.factory import create_azure_chat_llm
from app.config import Settings

import structlog

logger = structlog.get_logger(__name__)


def create_rag_chain(
    vectorstore: Chroma,
    settings: Settings,
    llm_model: str | None = None,
    temperature: float = 0.0,
) -> Runnable:
    """
    Create a RAG chain that retrieves from the menu vectorstore and answers questions.

    The chain is configured to only answer questions based on retrieved menu content.
    """
    llm = create_azure_chat_llm(settings, model=llm_model, temperature=temperature)

    # System prompt that enforces menu-only answers
    system_prompt = """You are a helpful barista assistant. Your job is to answer questions about the coffee menu based ONLY on the provided menu context.

Rules:
- Answer questions about drinks, prices, ingredients, and menu categories using the provided context.
- When asked to "show the menu" or "what's on the menu", provide a comprehensive overview of all available drinks from the context, including their prices and key details.
- If the information is not in the provided context, politely decline: "I'm sorry, I can only answer questions about our coffee menu. Could I help you with something from our menu instead?"
- Be friendly, concise, and accurate.
- Always include prices when mentioning drinks.
- Format menu listings clearly with drink names, prices, and brief descriptions."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nUse the following context to answer the question:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # Configure retriever with higher k to get more context for menu queries
    # We'll use k=10 to ensure we get all menu items for listing queries
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    # Create RAG chain using LCEL for better chat history support
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from operator import itemgetter

    def format_docs(docs):
        """Format documents, ensuring all content is string."""
        formatted = []
        for doc in docs:
            content = doc.page_content
            if isinstance(content, dict):
                content = str(content)
            elif not isinstance(content, str):
                content = str(content)
            formatted.append(content)
        return "\n\n".join(formatted)
    
    def normalize_question(question) -> str:
        """Normalize question to string, handling various input types."""
        if isinstance(question, dict):
            if "text" in question:
                return str(question["text"])
            elif "content" in question:
                return str(question["content"])
            else:
                return str(question)
        elif isinstance(question, list):
            text_parts = []
            for item in question:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        text_parts.append(str(item["text"]))
                    elif "content" in item:
                        text_parts.append(str(item["content"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts) if text_parts else str(question)
        elif not isinstance(question, str):
            return str(question)
        return question

    async def prepare_rag_inputs(inputs: dict) -> dict:
        """Prepare inputs for RAG chain, normalizing question and extracting values."""
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", [])

        # Normalize question to string - ensure it's definitely a string
        question_str = normalize_question(question)
        if not isinstance(question_str, str):
            question_str = str(question_str)

        # Retrieve documents using normalized question (async)
        # Retriever is configured with k=10 to get full menu content
        docs = await retriever.ainvoke(question_str)
        context_str = format_docs(docs)
        
        # Ensure context is a string
        if not isinstance(context_str, str):
            context_str = str(context_str)
        
        return {
            "context": context_str,
            "question": question_str,
            "chat_history": chat_history,
        }
    
    chain = (
        RunnableLambda(prepare_rag_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap to match expected interface
    class RAGChainWrapper:
        def __init__(self, chain, retriever):
            self.chain = chain
            self.retriever = retriever

        async def ainvoke(self, inputs):
            from langchain_core.messages import HumanMessage, AIMessage
            
            question = inputs.get("question", "")
            chat_history = inputs.get("chat_history", [])

            # Ensure question is a string (handle dict/list content) - use same logic as extract_question
            if isinstance(question, dict):
                if "text" in question:
                    question = str(question["text"])
                elif "content" in question:
                    question = str(question["content"])
                else:
                    question = str(question)
            elif isinstance(question, list):
                text_parts = []
                for item in question:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                        elif "text" in item:
                            text_parts.append(str(item["text"]))
                        elif "content" in item:
                            text_parts.append(str(item["content"]))
                    elif isinstance(item, str):
                        text_parts.append(item)
                question = " ".join(text_parts) if text_parts else str(question)
            elif not isinstance(question, str):
                question = str(question)
            
            # Final safety check - ensure question is definitely a string
            if not isinstance(question, str):
                logger.warning("rag.question_not_string", question_type=type(question).__name__)
                question = str(question)

            # Format chat history as actual message objects for MessagesPlaceholder
            history_messages = []
            for human, ai in chat_history:
                # Ensure content is a string
                human_str = str(human) if not isinstance(human, str) else human
                ai_str = str(ai) if not isinstance(ai, str) else ai
                history_messages.append(HumanMessage(content=human_str))
                history_messages.append(AIMessage(content=ai_str))

            # Get relevant documents for source attribution
            docs = await self.retriever.ainvoke(question)

            # Invoke chain - it will handle retriever and context formatting
            # Double-check that question is a string before passing
            chain_input = {
                "question": str(question),  # Force string conversion
                "chat_history": history_messages,
            }
            result = await self.chain.ainvoke(chain_input)

            return {
                "result": result,
                "source_documents": docs,
            }

    return RAGChainWrapper(chain, retriever)


async def query_menu(
    rag_chain: Runnable,
    question: str,
    chat_history: List[tuple[str, str]] | None = None,
) -> dict[str, str | List]:
    """
    Query the menu using RAG.

    Returns a dict with 'answer' and optionally 'sources'.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    logger.info("rag.query", question=question[:100])

    # Format chat history as actual message objects
    history_messages = []
    if chat_history:
        for human, ai in chat_history[-5:]:  # Last 5 exchanges
            human_str = str(human) if not isinstance(human, str) else human
            ai_str = str(ai) if not isinstance(ai, str) else ai
            history_messages.append(HumanMessage(content=human_str))
            history_messages.append(AIMessage(content=ai_str))

    result = await rag_chain.ainvoke(
        {
            "question": question,
            "chat_history": history_messages,
        }
    )

    answer = result.get("result", "I'm sorry, I couldn't find an answer to that question.")
    sources = result.get("source_documents", [])

    logger.info("rag.complete", answer_length=len(answer), sources_count=len(sources))

    return {
        "answer": answer,
        "sources": [doc.page_content[:200] for doc in sources] if sources else [],
    }

