"""Voice endpoints for Azure Speech Services (STT/TTS)."""

from __future__ import annotations

import base64
from io import BytesIO

import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])


class TranscribeRequest(BaseModel):
    """Transcribe request model."""

    audio_base64: str | None = Field(default=None, description="Base64 encoded audio data")
    language: str = Field(default="en-US", description="Language code for transcription")


class TranscribeResponse(BaseModel):
    """Transcribe response model."""

    text: str = Field(..., description="Transcribed text")
    language: str | None = Field(default=None, description="Detected language")


class SynthesizeRequest(BaseModel):
    """Text-to-speech request model."""

    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        default="en-US-JennyNeural",
        description="Voice name for synthesis",
    )
    language: str = Field(default="en-US", description="Language code")


class SynthesizeResponse(BaseModel):
    """Text-to-speech response model."""

    audio_base64: str = Field(..., description="Base64 encoded audio data")
    content_type: str = Field(default="audio/mpeg", description="Audio content type")


async def transcribe_audio(
    audio_data: bytes,
    speech_key: str,
    stt_endpoint: str,
    language: str = "en-US",
) -> str:
    """Transcribe audio using Azure Speech Services STT."""
    # Azure Speech Services STT endpoint
    # Ensure endpoint doesn't have trailing slash
    base_url = stt_endpoint.rstrip("/")
    url = f"{base_url}/speech/recognition/conversation/cognitiveservices/v1"
    params = {"language": language}

    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "audio/wav",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, params=params, headers=headers, content=audio_data)
            response.raise_for_status()
            data = response.json()

            if "RecognitionStatus" in data:
                if data["RecognitionStatus"] == "Success":
                    return data.get("DisplayText", data.get("Text", ""))
                else:
                    error_msg = data.get("RecognitionStatus", "Unknown error")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Speech recognition failed: {error_msg}",
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Unexpected response format from speech service",
                )
    except httpx.HTTPStatusError as e:
        logger.error("stt.http_error", status=e.response.status_code)
        error_detail = ""
        try:
            error_data = e.response.json()
            error_detail = str(error_data)
        except Exception:
            pass
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Speech recognition API error: {error_detail}",
        )
    except Exception as e:
        logger.error("stt.error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")


async def synthesize_text(
    text: str,
    speech_key: str,
    tts_endpoint: str,
    voice: str = "en-US-JennyNeural",
    language: str = "en-US",
) -> bytes:
    """Synthesize text to speech using Azure Speech Services TTS."""
    # Azure Speech Services TTS endpoint
    # Ensure endpoint doesn't have trailing slash
    base_url = tts_endpoint.rstrip("/")
    url = f"{base_url}/cognitiveservices/v1"

    # Create SSML for TTS
    ssml = f"""<speak version='1.0' xml:lang='{language}'>
    <voice xml:lang='{language}' name='{voice}'>
        {text}
    </voice>
</speak>"""

    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, content=ssml.encode("utf-8"))
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        logger.error("tts.http_error", status=e.response.status_code)
        error_detail = ""
        try:
            error_data = e.response.json()
            error_detail = str(error_data)
        except Exception:
            pass
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Text-to-speech API error: {error_detail}",
        )
    except Exception as e:
        logger.error("tts.error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")


@router.post("/transcribe", response_model=TranscribeResponse, summary="Transcribe audio to text")
async def transcribe(
    request: Request,
    audio_file: UploadFile | None = File(None),
    audio_base64: str | None = Form(None),
    language: str = Form("en-US"),
) -> TranscribeResponse:
    """
    Transcribe audio to text using Azure Speech Services STT.

    Accepts either:
    - Multipart form data with audio file
    - Form data with base64 encoded audio
    """
    ctx = request.app.state.ctx
    settings = ctx.settings

    if not settings.azure_speech_key:
        raise HTTPException(
            status_code=500,
            detail="Azure Speech Services not configured. Please set AZURE_SPEECH_KEY.",
        )

    # Get audio data
    audio_data = None
    if audio_file:
        audio_data = await audio_file.read()
    elif audio_base64:
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Either audio_file or audio_base64 must be provided",
        )

    if not audio_data:
        raise HTTPException(status_code=400, detail="No audio data provided")

    text = await transcribe_audio(
        audio_data=audio_data,
        speech_key=settings.azure_speech_key,
        stt_endpoint=settings.azure_speech_stt_endpoint,
        language=language,
    )

    logger.info("voice.transcribe.complete", text_length=len(text), language=language)

    return TranscribeResponse(text=text, language=language)


@router.post("/synthesize", response_model=SynthesizeResponse, summary="Synthesize text to speech")
async def synthesize(
    request: Request,
    synthesize_request: SynthesizeRequest,
) -> SynthesizeResponse:
    """
    Synthesize text to speech using Azure Speech Services TTS.

    Returns base64 encoded audio data.
    """
    ctx = request.app.state.ctx
    settings = ctx.settings

    if not settings.azure_speech_key:
        raise HTTPException(
            status_code=500,
            detail="Azure Speech Services not configured. Please set AZURE_SPEECH_KEY.",
        )

    audio_data = await synthesize_text(
        text=synthesize_request.text,
        speech_key=settings.azure_speech_key,
        tts_endpoint=settings.azure_speech_tts_endpoint,
        voice=synthesize_request.voice,
        language=synthesize_request.language,
    )

    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    logger.info(
        "voice.synthesize.complete",
        text_length=len(synthesize_request.text),
        voice=synthesize_request.voice,
    )

    return SynthesizeResponse(audio_base64=audio_base64, content_type="audio/mpeg")

