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

    # Only add image generation if Azure OpenAI API key is configured
    if ctx.settings.azure_openai_api_key:
        image_gen_tool = create_image_gen_tool(
            api_key=ctx.settings.azure_openai_api_key,
            endpoint=ctx.settings.azure_openai_endpoint,
            deployment_name=ctx.settings.azure_flux_deployment_name,
            api_version=ctx.settings.azure_flux_api_version,
        )
        tools.append(image_gen_tool)

    llm_with_tools = llm.bind_tools(tools)

    # Define graph nodes
    graph = StateGraph(AgentState)

    def route_decision(state: AgentState) -> Literal["tools", "rag"]:
        """Route to tools or RAG based on user intent."""
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()

            # Priority 1: Menu-related queries should go to RAG
            menu_keywords = ["menu", "what do you have", "what drinks", "what coffee", "offer", "selection", "list"]
            if any(keyword in content for keyword in menu_keywords):
                return "rag"

            # Priority 2: Check for tool-specific keywords
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

            # Priority 3: Image generation (but not for menu requests)
            if any(
                keyword in content
                for keyword in ["image", "picture", "photo", "looks like", "visual"]
            ):
                # Only route to tools if it's clearly about generating an image
                # and not about showing the menu
                if "menu" not in content and ctx.settings.azure_openai_api_key:
                    return "tools"
                # If menu is mentioned with image keywords, go to RAG
                return "rag"

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
            # Filter out large base64 image data to avoid context length issues
            tool_results_text_parts = []
            has_image = False
            for msg in tool_messages:
                content = msg.content
                # Handle dict content (from tool results)
                if isinstance(content, dict):
                    # If it's an image generation result, summarize it instead of including base64
                    if "image_url" in content:
                        image_url = content.get("image_url", "")
                        if image_url and image_url.startswith("data:image"):
                            # Truncate base64 data URL - just indicate image was generated
                            # The image URL is a data URL with base64, so we omit it to save tokens
                            tool_results_text_parts.append(
                                "- Image generated successfully"
                            )
                            has_image = True
                        else:
                            tool_results_text_parts.append(f"- Image URL: {image_url}")
                            has_image = True
                    elif "error" in content:
                        tool_results_text_parts.append(f"- Error: {content.get('error')}")
                    else:
                        # For other dict content, convert to string but limit size
                        content_str = str(content)
                        if len(content_str) > 500:
                            content_str = content_str[:500] + "... (truncated)"
                        tool_results_text_parts.append(f"- {content_str}")
                elif isinstance(content, str):
                    # For string content, check if it's a base64 data URL
                    if content.startswith("data:image") and len(content) > 1000:
                        # Truncate large base64 image data
                        tool_results_text_parts.append("- Image generated successfully (base64 data omitted)")
                        has_image = True
                    elif len(content) > 1000:
                        # Truncate other long strings
                        tool_results_text_parts.append(f"- {content[:500]}... (truncated)")
                    else:
                        tool_results_text_parts.append(f"- {content}")
                else:
                    # For other types, convert to string with size limit
                    content_str = str(content)
                    if len(content_str) > 500:
                        content_str = content_str[:500] + "... (truncated)"
                    tool_results_text_parts.append(f"- {content_str}")
            
            tool_results_text = "\n".join(tool_results_text_parts)
            
            # Special handling for image generation - tell LLM not to include image data
            image_instruction = ""
            if has_image:
                image_instruction = "\n\nIMPORTANT: Do NOT include any image URLs, base64 data, or markdown image syntax in your response. The image has been generated and will be displayed separately. Just mention that you've generated the image."
            
            response_prompt = f"""Based on the tool results below, provide a friendly, natural response to the user's question.

Tool Results:
{tool_results_text}

User's original question: {human_msg.content if human_msg else 'N/A'}
{image_instruction}

Provide a concise, helpful response that incorporates the tool results naturally. Be conversational and helpful. Do not include any technical details, URLs, or data in your response."""

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
