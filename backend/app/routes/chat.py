"""Chat endpoint for the Barista agent."""

from __future__ import annotations

import base64
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agent.graph import AgentState
from langchain_core.messages import HumanMessage

import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str | None = Field(default=None, description="User message (text)")
    audio_input: str | None = Field(
        default=None,
        description="Base64 encoded audio data for voice input (alternative to message)",
    )
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    audio_output: bool | None = Field(
        default=None,
        description="Whether to return audio response (defaults to TTS_ENABLED_BY_DEFAULT if not specified)",
    )


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    image_url: str | None = Field(default=None, description="Optional image URL if image was generated")
    audio_output: str | None = Field(
        default=None,
        description="Base64 encoded audio data if audio_output was requested",
    )


@router.post("/", response_model=ChatResponse, summary="Send a message to the Barista agent")
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    Process a chat message through the Barista agent.

    The agent will:
    - Answer menu questions using RAG
    - Check drink availability using the availability tool
    - Retrieve promotions using the promotions tool
    - Generate drink images using FLUX 1.1 (if configured)
    - Support voice input (transcribe audio) and voice output (synthesize response)

    Accepts either text message or audio_input (base64 encoded audio).
    If audio_output=true, returns synthesized audio response.
    """
    ctx = request.app.state.ctx
    agent_graph = ctx.agent_graph
    settings = ctx.settings

    # Generate or use session ID
    session_id = chat_request.session_id or str(uuid.uuid4())

    # Handle voice input: transcribe audio to text
    message_text = chat_request.message
    if chat_request.audio_input and not message_text:
        if not settings.azure_speech_key:
            raise HTTPException(
                status_code=400,
                detail="Audio input provided but Azure Speech Services not configured. Please set AZURE_SPEECH_KEY.",
            )
        try:
            from app.routes.voice import transcribe_audio

            audio_data = base64.b64decode(chat_request.audio_input)
            message_text = await transcribe_audio(
                audio_data=audio_data,
                speech_key=settings.azure_speech_key,
                stt_endpoint=settings.azure_speech_stt_endpoint,
            )
            logger.info("chat.voice_input.transcribed", session_id=session_id)
        except Exception as e:
            logger.error("chat.voice_input.error", error=str(e), session_id=session_id)
            raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {str(e)}")

    if not message_text:
        raise HTTPException(
            status_code=400,
            detail="Either message or audio_input must be provided",
        )

    # Create initial state
    # LangGraph's MemorySaver maintains conversation state via thread_id
    config = {"configurable": {"thread_id": session_id}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=message_text)],
        "session_id": session_id,
    }

    # Helper function to extract message content for logging
    def extract_message_content_for_logging(msg):
        """Extract string content from a message for logging."""
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(str(item) for item in content if item)
            else:
                return str(content)
        return str(msg)

    try:
        # Invoke agent graph
        # LangGraph's MemorySaver maintains conversation state via thread_id (session_id)
        # Each invocation with the same thread_id loads previous state and appends new messages
        # This provides conversation continuity
        logger.info(
            "chat.invoking_agent",
            session_id=session_id,
            message=message_text[:100],  # Log first 100 chars
        )
        result = await agent_graph.ainvoke(initial_state, config)
        
        # Log the number of messages in the result to verify state is being maintained
        # If this number keeps growing, it means state is being maintained correctly
        message_count = len(result.get("messages", []))
        recent_messages = [
            extract_message_content_for_logging(msg)[:50] 
            for msg in result.get("messages", [])[-4:]
        ]
        logger.info(
            "chat.agent_result",
            session_id=session_id,
            message_count=message_count,
            recent_messages=recent_messages,
        )

        # Extract final AI message
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        if not ai_messages:
            raise HTTPException(status_code=500, detail="Agent did not generate a response")

        final_message = ai_messages[-1]
        # Extract content from message, handling various content types
        # AIMessage.content can be str, list, or dict
        content = final_message.content
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            # Handle list of content blocks (e.g., from tool calls)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            response_text = " ".join(text_parts) if text_parts else ""
        elif isinstance(content, dict):
            if "text" in content:
                response_text = str(content["text"])
            else:
                response_text = str(content)
        else:
            response_text = str(content) if content else ""
        
        if not response_text:
            raise HTTPException(status_code=500, detail="Agent response is empty")

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

        # Handle voice output: synthesize text to speech if requested
        # Use config default if audio_output not explicitly set
        should_synthesize = (
            chat_request.audio_output
            if chat_request.audio_output is not None
            else settings.tts_enabled_by_default
        )

        audio_output_base64 = None
        if should_synthesize:
            if not settings.azure_speech_key:
                logger.warning("chat.audio_output.requested_but_not_configured", session_id=session_id)
            else:
                try:
                    from app.routes.voice import synthesize_text

                    audio_data = await synthesize_text(
                        text=response_text,
                        speech_key=settings.azure_speech_key,
                        tts_endpoint=settings.azure_speech_tts_endpoint,
                        voice=settings.tts_voice,
                        language=settings.tts_language,
                    )
                    audio_output_base64 = base64.b64encode(audio_data).decode("utf-8")
                    logger.info("chat.audio_output.synthesized", session_id=session_id)
                except Exception as e:
                    logger.error("chat.audio_output.error", error=str(e), session_id=session_id)
                    # Don't fail the request if TTS fails, just log it

        logger.info(
            "chat.complete",
            session_id=session_id,
            message_length=len(message_text),
            response_length=len(response_text),
            has_image=image_url is not None,
            has_audio_output=audio_output_base64 is not None,
        )

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            image_url=image_url,
            audio_output=audio_output_base64,
        )

    except Exception as e:
        logger.error("chat.error", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

