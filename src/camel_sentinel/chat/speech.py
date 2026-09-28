"""
chat/speech.py — Azure Cognitive Services Speech: STT (Slice C) + TTS (Slice D).

Both functions share the same _speech_config() helper and the same two env
vars: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (see .env.example).

STT (transcribe_audio): accepts WAV bytes from the browser's audio recorder,
writes them to a temp file, and runs Azure recognize_once(). Slice C.

TTS (synthesize_text): converts a text reply to WAV bytes using Azure
SpeechSynthesizer. Added in Slice D — placeholder import guard below.

WHY temp file instead of a push stream: recognize_once() with AudioConfig
(filename=...) is simpler, stateless, and sufficient for the single-utterance
request pattern the /speech/transcribe endpoint uses. A push stream would be
needed only for real-time streaming recognition, which is not required here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _speech_config():
    """Return an Azure SpeechConfig; raise EnvironmentError if keys are missing."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise ImportError(
            "azure-cognitiveservices-speech is not installed. "
            "It should already be present — check the project venv."
        ) from exc

    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        raise EnvironmentError(
            "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set. "
            "See .env.example."
        )
    return speechsdk.SpeechConfig(subscription=key, region=region)


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert WAV audio bytes to text using Azure Speech-to-Text.

    Parameters
    ----------
    audio_bytes : bytes
        WAV audio from the browser (st.audio_input returns 16 kHz WAV).

    Returns
    -------
    str
        Recognised text, or empty string if nothing was understood.

    Raises
    ------
    RuntimeError
        If Azure returns a Canceled result (bad key, network error, etc.).
    EnvironmentError
        If AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set.
    """
    import azure.cognitiveservices.speech as speechsdk

    config = _speech_config()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    audio_config = None
    recognizer = None
    try:
        audio_config = speechsdk.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=config,
            audio_config=audio_config,
        )
        result = recognizer.recognize_once()
    finally:
        # On Windows the SDK holds a native file handle until the objects are
        # released. Set to None to drop the reference (CPython ref-counting
        # destructs immediately) so the handle closes before unlink;
        # otherwise Windows raises WinError 32 (file in use by another process).
        recognizer = None
        audio_config = None
        Path(tmp_path).unlink(missing_ok=True)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    if result.reason == speechsdk.ResultReason.NoMatch:
        return ""
    if result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails.from_result(result)
        raise RuntimeError(
            f"Azure Speech recognition cancelled: {details.reason}. "
            f"Error details: {details.error_details}"
        )
    return ""


# ---------------------------------------------------------------------------
# TTS — Slice D
# ---------------------------------------------------------------------------

def synthesize_text(text: str, voice: str = "en-US-AriaNeural") -> bytes:
    """
    Convert text to WAV audio bytes using Azure Text-to-Speech.

    WHY temp file instead of a pull stream: AudioOutputConfig(filename=...)
    is stateless and simpler to reason about than managing a pull-stream
    cursor across threads. The file is deleted in the finally block.

    Parameters
    ----------
    text : str
        The reply text to speak.
    voice : str
        Azure Neural voice name.  Default is en-US-AriaNeural.

    Returns
    -------
    bytes
        WAV audio bytes ready for st.audio() or an HTTP response.

    Raises
    ------
    RuntimeError
        If Azure returns a Canceled result.
    EnvironmentError
        If AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set.
    """
    import azure.cognitiveservices.speech as speechsdk

    config = _speech_config()
    config.speech_synthesis_voice_name = voice

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    audio_output = None
    synthesizer = None
    try:
        audio_output = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config,
            audio_config=audio_output,
        )
        result = synthesizer.speak_text_async(text).get()

        # Release SDK objects before reading/deleting the file — on Windows the
        # synthesizer holds a native write handle until its reference drops to 0.
        synthesizer = None
        audio_output = None

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            with open(tmp_path, "rb") as f:
                return f.read()

        if result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.CancellationDetails.from_result(result)
            raise RuntimeError(
                f"Azure TTS cancelled: {details.reason}. "
                f"Error details: {details.error_details}"
            )
        raise RuntimeError(f"Azure TTS unexpected result: {result.reason}")
    finally:
        synthesizer = None
        audio_output = None
        Path(tmp_path).unlink(missing_ok=True)
