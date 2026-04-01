"""
🎙️ Terminal-based Thai Voice → English Text Translator
Uses PyAudio for microphone capture + Gemini Live API.

Usage:
    python app_terminal.py                  # Always Listening (default)
    python app_terminal.py --push-to-talk   # Push-to-Talk (Enter to record, Enter to stop)
"""
import asyncio
import os
import sys
import threading
import queue
import argparse
import time

from dotenv import load_dotenv

load_dotenv()


def record_audio_continuous(audio_queue: queue.Queue, stop_event: threading.Event):
    """Record audio continuously from microphone."""
    import pyaudio

    RATE = 16000
    CHUNK = 1600

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16, channels=1, rate=RATE,
        input=True, frames_per_buffer=CHUNK,
    )
    print("🔴 Microphone LIVE! Start speaking Thai...")
    print()

    try:
        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_queue.put(data)
            except OSError:
                pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def record_audio_push_to_talk(audio_queue: queue.Queue, stop_event: threading.Event,
                               recording_event: threading.Event):
    """Record audio only when recording_event is set."""
    import pyaudio

    RATE = 16000
    CHUNK = 1600

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16, channels=1, rate=RATE,
        input=True, frames_per_buffer=CHUNK,
    )

    try:
        while not stop_event.is_set():
            if recording_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_queue.put(data)
                except OSError:
                    pass
            else:
                time.sleep(0.05)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def run_always_listening():
    """Always Listening mode — uses async with for Gemini session."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env!")
        sys.exit(1)

    print("=" * 60)
    print("🎙️  Thai Voice → English Text | ALWAYS LISTENING")
    print("=" * 60)
    print()
    print("🔗 Connecting to Gemini Live API...")

    client = genai.Client(api_key=api_key)

    from gemini_client import _build_config
    config = _build_config()

    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    async with client.aio.live.connect(model="gemini-3.1-flash-live-preview", config=config) as session:
        print("✅ Connected!")
        print("📢 Mode: Always Listening (auto voice detection)")
        print("💡 พูดภาษาไทยได้เลย... (Ctrl+C to stop)")
        print("-" * 60)
        print()

        mic_thread = threading.Thread(
            target=record_audio_continuous,
            args=(audio_queue, stop_event),
            daemon=True,
        )
        mic_thread.start()

        async def send_audio():
            while not stop_event.is_set():
                try:
                    data = audio_queue.get(timeout=0.1)
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    if not stop_event.is_set():
                        print(f"\n❌ Send error: {e}")
                    break

        async def receive_responses():
            current_thai = ""
            current_english = ""
            try:
                async for response in session.receive():
                    if stop_event.is_set():
                        break
                    if response.server_content:
                        content = response.server_content
                        if content.input_transcription:
                            t = content.input_transcription.text
                            if t and t.strip():
                                current_thai += t
                                print(f"\r🇹🇭 TH: {current_thai}", end="", flush=True)
                        if content.output_transcription:
                            t = content.output_transcription.text
                            if t and t.strip():
                                current_english += t
                                print(f"\n🇬🇧 EN: {current_english}", end="", flush=True)
                        if content.model_turn:
                            for part in content.model_turn.parts:
                                if part.text and part.text.strip():
                                    print(f"\n🇬🇧 EN: {part.text.strip()}", flush=True)
                        if content.turn_complete:
                            if current_thai or current_english:
                                print()
                                print("-" * 40)
                            current_thai = ""
                            current_english = ""
            except Exception as e:
                if not stop_event.is_set():
                    print(f"\n❌ Receive error: {e}")

        try:
            await asyncio.gather(send_audio(), receive_responses())
        except KeyboardInterrupt:
            pass
        finally:
            print("\n\n🛑 Stopping...")
            stop_event.set()
            mic_thread.join(timeout=2)
            print("👋 Goodbye!")


async def run_push_to_talk():
    """Push-to-Talk mode — uses async with for Gemini session."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env!")
        sys.exit(1)

    print("=" * 60)
    print("🎙️  Thai Voice → English Text | PUSH-TO-TALK")
    print("=" * 60)
    print()
    print("🔗 Connecting to Gemini Live API...")

    client = genai.Client(api_key=api_key)

    from gemini_client import _build_config
    config = _build_config()

    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    recording_event = threading.Event()

    async with client.aio.live.connect(model="gemini-3.1-flash-live-preview", config=config) as session:
        print("✅ Connected!")
        print("📢 Mode: Push-to-Talk")
        print("💡 กด Enter เพื่อเริ่มพูด, กด Enter อีกครั้งเพื่อหยุด")
        print("💡 พิมพ์ 'q' เพื่อออก")
        print("-" * 60)
        print()

        mic_thread = threading.Thread(
            target=record_audio_push_to_talk,
            args=(audio_queue, stop_event, recording_event),
            daemon=True,
        )
        mic_thread.start()

        async def send_audio():
            while not stop_event.is_set():
                try:
                    data = audio_queue.get(timeout=0.1)
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    if not stop_event.is_set():
                        print(f"\n❌ Send error: {e}")
                    break

        async def receive_responses():
            current_thai = ""
            current_english = ""
            try:
                async for response in session.receive():
                    if stop_event.is_set():
                        break
                    if response.server_content:
                        content = response.server_content
                        if content.input_transcription:
                            t = content.input_transcription.text
                            if t and t.strip():
                                current_thai += t
                                print(f"\r🇹🇭 TH: {current_thai}", end="", flush=True)
                        if content.output_transcription:
                            t = content.output_transcription.text
                            if t and t.strip():
                                current_english += t
                                print(f"\n🇬🇧 EN: {current_english}", end="", flush=True)
                        if content.model_turn:
                            for part in content.model_turn.parts:
                                if part.text and part.text.strip():
                                    print(f"\n🇬🇧 EN: {part.text.strip()}", flush=True)
                        if content.turn_complete:
                            if current_thai or current_english:
                                print()
                                print("-" * 40)
                            current_thai = ""
                            current_english = ""
            except Exception as e:
                if not stop_event.is_set():
                    print(f"\n❌ Receive error: {e}")

        async def user_control():
            loop = asyncio.get_event_loop()
            is_recording = False
            while not stop_event.is_set():
                prompt = ("🔴 RECORDING... Enter=หยุด / q=ออก: " if is_recording
                          else "⚪ Enter=เริ่มพูด / q=ออก: ")
                try:
                    user_input = await loop.run_in_executor(None, lambda: input(prompt))
                except EOFError:
                    break
                if user_input.strip().lower() == 'q':
                    stop_event.set()
                    break
                if is_recording:
                    recording_event.clear()
                    is_recording = False
                    await session.send_realtime_input(audio_stream_end=True)
                    print("⏹ หยุดบันทึกเสียง\n")
                else:
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    recording_event.set()
                    is_recording = True
                    print("🔴 เริ่มบันทึก... พูดภาษาไทยได้เลย!")

        try:
            await asyncio.gather(send_audio(), receive_responses(), user_control())
        except KeyboardInterrupt:
            pass
        finally:
            print("\n\n🛑 Stopping...")
            stop_event.set()
            mic_thread.join(timeout=2)
            print("👋 Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thai Voice → English Text Translator")
    parser.add_argument(
        "--push-to-talk", "-p",
        action="store_true",
        help="Use Push-to-Talk mode (default: Always Listening)"
    )
    args = parser.parse_args()

    try:
        if args.push_to_talk:
            asyncio.run(run_push_to_talk())
        else:
            asyncio.run(run_always_listening())
    except KeyboardInterrupt:
        print("\n👋 Bye!")
