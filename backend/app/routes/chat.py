"""Chat endpoint for the Barista agent."""

from __future__ import annotations

import base64
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

    message: str | None = Field(default=None, description="User message (text)")
    audio_input: str | None = Field(
        default=None,
        description="Base64 encoded audio data for voice input (alternative to message)",
    )
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    audio_output: bool = Field(default=False, description="Whether to return audio response")


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
    config = {"configurable": {"thread_id": session_id}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=message_text)],
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

        # Handle voice output: synthesize text to speech if requested
        audio_output_base64 = None
        if chat_request.audio_output:
            if not settings.azure_speech_key:
                logger.warning("chat.audio_output.requested_but_not_configured", session_id=session_id)
            else:
                try:
                    from app.routes.voice import synthesize_text

                    audio_data = await synthesize_text(
                        text=response_text,
                        speech_key=settings.azure_speech_key,
                        tts_endpoint=settings.azure_speech_tts_endpoint,
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

