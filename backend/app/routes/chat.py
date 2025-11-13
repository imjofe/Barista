"""Chat endpoint for the Barista agent."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agent.graph import AgentState
from langchain_core.messages import HumanMessage

import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., description="User message")
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    image_url: str | None = Field(default=None, description="Optional image URL if image was generated")


@router.post("/", response_model=ChatResponse, summary="Send a message to the Barista agent")
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    Process a chat message through the Barista agent.

    The agent will:
    - Answer menu questions using RAG
    - Check drink availability using the availability tool
    - Retrieve promotions using the promotions tool
    - Generate drink images using the image generation tool (if configured)
    """
    ctx = request.app.state.ctx
    agent_graph = ctx.agent_graph

    # Generate or use session ID
    session_id = chat_request.session_id or str(uuid.uuid4())

    # Create initial state
    config = {"configurable": {"thread_id": session_id}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=chat_request.message)],
        "session_id": session_id,
    }

    try:
        # Invoke agent graph
        result = await agent_graph.ainvoke(initial_state, config)

        # Extract final AI message
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        if not ai_messages:
            raise HTTPException(status_code=500, detail="Agent did not generate a response")

        final_message = ai_messages[-1]
        response_text = final_message.content

        # Check for image URL in tool results (if image generation was used)
        image_url = None
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        for tool_msg in tool_messages:
            # Try to parse tool result for image URL
            if hasattr(tool_msg, "content"):
                content = tool_msg.content
                if isinstance(content, dict) and "image_url" in content:
                    image_url = content.get("image_url")
                    break
                elif isinstance(content, str) and "image_url" in content:
                    # Try to extract from string
                    import json
                    try:
                        parsed = json.loads(content)
                        if "image_url" in parsed:
                            image_url = parsed["image_url"]
                    except (json.JSONDecodeError, ValueError):
                        pass

        logger.info(
            "chat.complete",
            session_id=session_id,
            message_length=len(chat_request.message),
            response_length=len(response_text),
            has_image=image_url is not None,
        )

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            image_url=image_url,
        )

    except Exception as e:
        logger.error("chat.error", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

