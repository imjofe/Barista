"""LangGraph workflow construction for the Barista agent."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.config import AppContext
from app.knowledge.rag import create_rag_chain
from app.llm.factory import create_azure_chat_llm
from app.tools import check_drink_availability, get_daily_promotion, create_image_gen_tool

import structlog

logger = structlog.get_logger(__name__)


class AgentState(TypedDict):
    """State schema for the Barista agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str


def build_agent_graph(ctx: AppContext) -> Runnable:
    """
    Construct the Barista agent LangGraph.

    The graph implements:
    1. Intent routing (RAG vs tools)
    2. Tool execution (availability, promotions, image generation)
    3. Policy enforcement (out-of-scope rejection)
    """
    from app.knowledge.ingestion import ingest_menu_to_chroma

    # Initialize vectorstore
    # Try multiple paths for menu.md
    menu_path = None
    possible_paths = [
        Path("/app/menu.md"),  # Docker image location (first priority)
        Path(__file__).parent.parent / "menu.md",  # Backend root
        Path(ctx.settings.chroma_persist_path).parent / "menu.md",
    ]
    
    for path in possible_paths:
        try:
            # Check if path exists and is a file
            if path.exists():
                if path.is_file():
                    # Try to read first byte to verify we have read access
                    try:
                        with open(path, "rb") as f:
                            f.read(1)
                        menu_path = path
                        logger.info("menu_found", path=str(path))
                        break
                    except (PermissionError, IOError) as e:
                        logger.warning("menu_no_read_access", path=str(path), error=str(e))
                        continue
        except (PermissionError, OSError) as e:
            # Skip paths we can't access (e.g., parent directory permissions)
            logger.debug("menu_path_check_failed", path=str(path), error=str(e))
            continue
    
    if menu_path is None:
        raise FileNotFoundError(f"menu.md not found or not accessible. Tried: {[str(p) for p in possible_paths]}")

    vectorstore = ingest_menu_to_chroma(
        menu_path=menu_path,
        persist_path=ctx.settings.chroma_persist_path,
        settings=ctx.settings,
    )

    # Create RAG chain
    rag_chain = create_rag_chain(
        vectorstore=vectorstore,
        settings=ctx.settings,
    )

    # Initialize LLM with tools
    llm = create_azure_chat_llm(ctx.settings, temperature=0.0)

    # Bind tools to LLM
    tools = [
        check_drink_availability,
        get_daily_promotion,
    ]

    # Only add image generation if API key is configured
    if ctx.settings.stability_api_key:
        image_gen_tool = create_image_gen_tool(ctx.settings.stability_api_key)
        tools.append(image_gen_tool)

    llm_with_tools = llm.bind_tools(tools)

    # Define graph nodes
    graph = StateGraph(AgentState)

    def route_decision(state: AgentState) -> Literal["tools", "rag"]:
        """Route to tools or RAG based on user intent."""
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()

            # Check for tool-specific keywords
            if any(
                keyword in content
                for keyword in ["available", "availability", "when", "time", "now"]
            ):
                # Check if asking about a specific drink availability
                drinks = [
                    "mocha magic",
                    "vanilla dream",
                    "caramel delight",
                    "hazelnut harmony",
                    "espresso elixir",
                    "latte lux",
                    "cappuccino charm",
                ]
                if any(drink in content for drink in drinks):
                    return "tools"

            if any(
                keyword in content
                for keyword in ["promotion", "special", "deal", "discount", "offer"]
            ):
                return "tools"

            if any(
                keyword in content
                for keyword in ["show", "image", "picture", "photo", "looks like", "visual"]
            ):
                if ctx.settings.stability_api_key:
                    return "tools"
                # Fall through to RAG if image gen not available

            # Default to RAG for menu questions
            return "rag"

        return "rag"
    
    def router(state: AgentState) -> AgentState:
        """Router node - passes through state unchanged."""
        # The routing decision is made by route_decision function
        return state

    def extract_message_content(message: BaseMessage) -> str:
        """Extract string content from a message, handling various content types."""
        content = message.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle list of content blocks (e.g., from tool calls)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    # Extract text from dict blocks
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts) if text_parts else ""
        elif isinstance(content, dict):
            # Handle dict content
            if "text" in content:
                return str(content["text"])
            return str(content)
        else:
            return str(content) if content else ""

    async def call_rag(state: AgentState) -> AgentState:
        """Invoke RAG chain for menu questions."""
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            question = extract_message_content(last_message)

            # Get chat history from messages
            chat_history = []
            for i in range(0, len(state["messages"]) - 1):
                msg = state["messages"][i]
                # Look for pairs of HumanMessage followed by AIMessage
                if isinstance(msg, HumanMessage):
                    # Find the next AIMessage (might not be immediately after)
                    for j in range(i + 1, len(state["messages"]) - 1):
                        next_msg = state["messages"][j]
                        if isinstance(next_msg, AIMessage):
                            human_content = extract_message_content(msg)
                            ai_content = extract_message_content(next_msg)
                            # Only add if both have valid string content
                            if human_content and ai_content:
                                chat_history.append((human_content, ai_content))
                            break

            result = await rag_chain.ainvoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            )

            answer = result.get("result", "I'm sorry, I couldn't find an answer to that question.")
            if not isinstance(answer, str):
                answer = str(answer)

            state["messages"].append(AIMessage(content=answer))
        return state

    async def call_llm_with_tools(state: AgentState) -> AgentState:
        """Invoke LLM with tool binding for tool selection."""
        response = await llm_with_tools.ainvoke(state["messages"])
        state["messages"].append(response)
        return state

    # Create tool node (shared instance)
    tool_node = ToolNode(tools)

    async def finalize_response(state: AgentState) -> AgentState:
        """Finalize response, handling tool results."""
        # Check if we have tool results that need to be converted to natural language
        tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
        
        if tool_messages:
            # Get the last human message (before tool calls)
            human_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    human_msg = msg
                    break

            # Generate natural language response from tool results
            tool_results_text = "\n".join([f"- {msg.content}" for msg in tool_messages])
            response_prompt = f"""Based on the tool results below, provide a friendly, natural response to the user's question.

Tool Results:
{tool_results_text}

User's original question: {human_msg.content if human_msg else 'N/A'}

Provide a concise, helpful response that incorporates the tool results naturally. Be conversational and helpful."""

            final_response = await llm.ainvoke([HumanMessage(content=response_prompt)])
            state["messages"].append(AIMessage(content=final_response.content))

        return state

    # Add nodes
    graph.add_node("router", router)
    graph.add_node("llm", call_llm_with_tools)
    graph.add_node("rag", call_rag)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize_response)

    # Set entry point
    graph.set_entry_point("router")

    # Add routing from router
    graph.add_conditional_edges(
        "router",
        route_decision,  # Use route_decision function for routing logic
        {
            "tools": "llm",  # Use LLM to decide which tool
            "rag": "rag",
        },
    )

    # After LLM, check if tools were called
    def should_call_tools(state: AgentState) -> Literal["tools", "finalize"]:
        """Check if LLM response contains tool calls."""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "finalize"
    
    graph.add_conditional_edges(
        "llm",
        should_call_tools,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )

    # After tools, finalize
    graph.add_edge("tools", "finalize")
    graph.add_edge("rag", "finalize")
    graph.add_edge("finalize", END)

    # Add memory/checkpointing
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app
