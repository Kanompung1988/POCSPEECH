"""
Gemini Live API Client
Real-time Thai Voice → English Text translation via WebSocket.

Uses `async with` for the session lifecycle (required by google-genai SDK).
The send/recv loops run forever until stop_event is set.
"""
import asyncio
import logging
import queue
from typing import Optional

from google import genai
from google.genai import types

log = logging.getLogger("gemini_live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


SYSTEM_INSTRUCTION = """You are a real-time Thai to English translator.

RULES:
1. When you hear Thai speech, translate it to English immediately.
2. Output ONLY the English translation. No explanations, no formatting.
3. If the speech is unclear, output your best translation attempt.
4. Keep translations natural and conversational.
5. If you hear English, just repeat it as-is.
6. Do not add quotes, labels, or prefixes.
7. Translate each sentence as soon as you hear it.
"""


def _build_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_INSTRUCTION)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"
                )
            )
        ),
        # ── Aggressive VAD: cut turns FAST for real-time feel ──
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                # Detect start of speech quickly
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                # Detect end of speech quickly → shorter turns
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                # Minimum silence before turn ends (ms) — as low as possible
                silence_duration_ms=100,
                # Minimum speech before confirming start (ms) — quick start
                prefix_padding_ms=20,
            )
        ),
    )


async def run_live_session(
    api_key: str,
    audio_in: queue.Queue,
    result_out: queue.Queue,
    stop_event: asyncio.Event,
    model: str = "gemini-3.1-flash-live-preview",
):
    """
    Full lifecycle: connect → send audio / receive results → close.
    Both _send and _recv loop forever until stop_event is set.
    """
    client = genai.Client(api_key=api_key)
    config = _build_config()

    async with client.aio.live.connect(model=model, config=config) as session:
        result_out.put({"status": "connected"})
        log.info("✅ Connected to Gemini Live session")

        async def _send():
            """Send audio from queue to Gemini. Never exits unless stopped."""
            loop = asyncio.get_running_loop()
            while not stop_event.is_set():
                try:
                    # Use run_in_executor so blocking queue.get doesn't block the event loop
                    data = await loop.run_in_executor(
                        None, lambda: audio_in.get(timeout=0.1)
                    )
                except queue.Empty:
                    continue

                try:
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                except Exception as e:
                    if not stop_event.is_set():
                        result_out.put({"error": f"Send: {e}"})
                    return  # session is broken, exit

        async def _recv():
            """Receive responses from Gemini. Never exits unless stopped.

            IMPORTANT: session.receive() yields messages for ONE model turn
            only, then the async iterator ends. We must loop and call
            session.receive() again for each subsequent turn.
            """
            turn_num = 0
            while not stop_event.is_set():
                turn_num += 1
                log.info(f"👂 Waiting for turn #{turn_num}...")
                try:
                    async for response in session.receive():
                        if stop_event.is_set():
                            return

                        if not response.server_content:
                            continue

                        content = response.server_content

                        # Input transcription (Thai)
                        if content.input_transcription:
                            t = content.input_transcription.text
                            if t:
                                log.info(f"📝 TH chunk: '{t}' (finished={content.input_transcription.finished})")
                                result_out.put({"thai": t})

                        # Output transcription (English translation)
                        if content.output_transcription:
                            t = content.output_transcription.text
                            if t:
                                log.info(f"📝 EN chunk: '{t}' (finished={content.output_transcription.finished})")
                                result_out.put({"english": t})

                        # Model turn: may contain inline audio data + text
                        if content.model_turn:
                            for part in content.model_turn.parts:
                                if part.text:
                                    result_out.put({"english": part.text})

                        # Turn complete marker
                        if content.turn_complete:
                            result_out.put({"turn_complete": True})
                            log.info(f"🔄 Turn #{turn_num} complete")

                except Exception as e:
                    if not stop_event.is_set():
                        log.error(f"❌ _recv exception on turn #{turn_num}: {e}")
                        result_out.put({"error": f"Recv: {e}"})
                        return  # broken session, exit

        # Run both forever — first one to return (error/stop) cancels the other
        done, pending = await asyncio.wait(
            [asyncio.create_task(_send()), asyncio.create_task(_recv())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel whatever is still running
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Log which task finished first
        for task in done:
            log.info(f"🛑 Task finished: {task.get_name()} result={task.exception() or 'ok'}")

    log.info("🔌 Session closed")
    result_out.put({"status": "disconnected"})
