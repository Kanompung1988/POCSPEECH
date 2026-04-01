"""
🎙️ Real-time Thai Voice → English Text
FastAPI WebSocket + HTML/JS for true streaming subtitles.
Text flows word-by-word as Gemini returns chunks.

Usage:
    python app_realtime.py
    Open http://localhost:8000 in browser
"""
import asyncio
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from gemini_client import run_live_session

load_dotenv()

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI()

# ─── Global state (single-user PoC) ───────────────────────
audio_queue: queue.Queue = queue.Queue()
result_queue: queue.Queue = queue.Queue()
stop_event: threading.Event = threading.Event()
gemini_thread: threading.Thread | None = None
mic_thread: threading.Thread | None = None
is_running = False


# ─── Gemini worker ────────────────────────────────────────
def gemini_worker():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        result_queue.put({"error": "GEMINI_API_KEY not set"})
        return

    async def _run():
        async_stop = asyncio.Event()

        async def _watch():
            while not stop_event.is_set():
                await asyncio.sleep(0.05)
            async_stop.set()

        asyncio.create_task(_watch())

        try:
            await run_live_session(
                api_key=api_key,
                audio_in=audio_queue,
                result_out=result_queue,
                stop_event=async_stop,
            )
        except Exception as e:
            result_queue.put({"error": str(e)})

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


# ─── Mic worker ───────────────────────────────────────────
def mic_worker_fn():
    import pyaudio

    RATE = 16000
    CHUNK = 800  # 50ms

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    log.info("🎙️ Mic started")
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
        log.info("🎙️ Mic stopped")


# ─── Start / Stop ─────────────────────────────────────────
def start_all():
    global gemini_thread, mic_thread, is_running, audio_queue, result_queue, stop_event
    if is_running:
        return
    stop_event.clear()
    # Flush queues
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: pass
    while not result_queue.empty():
        try: result_queue.get_nowait()
        except: pass

    gemini_thread = threading.Thread(target=gemini_worker, daemon=True)
    gemini_thread.start()

    mic_thread = threading.Thread(target=mic_worker_fn, daemon=True)
    mic_thread.start()

    is_running = True
    log.info("▶ Session started")


def stop_all():
    global is_running, gemini_thread, mic_thread
    stop_event.set()
    if gemini_thread:
        gemini_thread.join(timeout=3)
    if mic_thread:
        mic_thread.join(timeout=3)
    gemini_thread = None
    mic_thread = None
    is_running = False
    log.info("⏹ Session stopped")


# ─── WebSocket: push chunks to browser in real-time ───────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("🌐 Browser connected via WebSocket")

    try:
        # Wait for start command from browser
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data.get("action") == "start":
                start_all()

                # Wait for Gemini connected
                for _ in range(100):
                    try:
                        r = result_queue.get(timeout=0.05)
                        if r.get("status") == "connected":
                            await ws.send_json({"type": "status", "value": "connected"})
                            break
                        if "error" in r:
                            await ws.send_json({"type": "error", "value": r["error"]})
                            return
                    except queue.Empty:
                        pass

                # Stream results to browser as fast as they come
                while is_running and not stop_event.is_set():
                    try:
                        r = result_queue.get(timeout=0.02)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if "thai" in r:
                        await ws.send_json({"type": "thai", "text": r["thai"]})
                    if "english" in r:
                        await ws.send_json({"type": "english", "text": r["english"]})
                    if r.get("turn_complete"):
                        await ws.send_json({"type": "turn_complete"})
                    if "error" in r:
                        await ws.send_json({"type": "error", "value": r["error"]})
                    if r.get("status") == "disconnected":
                        await ws.send_json({"type": "status", "value": "disconnected"})
                        break

            elif data.get("action") == "stop":
                stop_all()
                await ws.send_json({"type": "status", "value": "stopped"})

    except WebSocketDisconnect:
        log.info("🌐 Browser disconnected")
        stop_all()
    except Exception as e:
        log.error(f"WS error: {e}")
        stop_all()


# ─── HTML page with real-time JS ──────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎙️ Thai Voice → English Text</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    min-height: 100vh;
  }

  .header {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 0 0 20px 20px;
  }
  .header h1 { font-size: 1.8rem; color: white; }
  .header p { color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 4px; }

  .controls {
    display: flex;
    gap: 12px;
    justify-content: center;
    padding: 1.2rem;
  }
  .controls button {
    padding: 10px 32px;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  #btnStart {
    background: #00b894;
    color: white;
  }
  #btnStart:hover { background: #00a381; }
  #btnStart:disabled { background: #444; color: #888; cursor: not-allowed; }
  #btnStop {
    background: #d63031;
    color: white;
  }
  #btnStop:hover { background: #b52526; }
  #btnStop:disabled { background: #444; color: #888; cursor: not-allowed; }
  #btnClear {
    background: #2d3436;
    color: #dfe6e9;
    border: 1px solid #636e72;
  }
  #btnClear:hover { background: #3d4446; }

  .status {
    text-align: center;
    padding: 6px;
    font-size: 0.85rem;
  }
  .status.idle { color: #636e72; }
  .status.connected { color: #55efc4; }
  .status.recording { color: #ff6b6b; }

  .transcript-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 1.2rem 1.2rem;
  }
  .transcript-label {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: #b2bec3;
  }

  #transcript {
    background: #12121a;
    border: 1px solid #2d2d3d;
    border-radius: 16px;
    padding: 1.2rem;
    min-height: 60vh;
    max-height: 75vh;
    overflow-y: auto;
    scroll-behavior: smooth;
  }

  .turn-block {
    margin-bottom: 12px;
    animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .th-line {
    color: #74b9ff;
    font-size: 0.95rem;
    padding: 6px 12px;
    margin: 2px 0;
    border-left: 3px solid #74b9ff;
    background: rgba(116, 185, 255, 0.06);
    border-radius: 0 8px 8px 0;
    min-height: 1.4em;
  }

  .en-line {
    color: #55efc4;
    font-size: 1.1rem;
    font-weight: 500;
    padding: 6px 12px;
    margin: 2px 0;
    border-left: 3px solid #55efc4;
    background: rgba(85, 239, 196, 0.06);
    border-radius: 0 8px 8px 0;
    min-height: 1.4em;
  }

  /* Blinking cursor for active line */
  .active::after {
    content: '▊';
    animation: blink 0.8s infinite;
    color: inherit;
    opacity: 0.6;
  }
  @keyframes blink {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 0; }
  }

  .divider {
    border: none;
    border-top: 1px solid #2d2d3d;
    margin: 8px 0;
  }

  .footer {
    text-align: center;
    padding: 1rem;
    color: #636e72;
    font-size: 0.8rem;
  }

  /* Scrollbar */
  #transcript::-webkit-scrollbar { width: 6px; }
  #transcript::-webkit-scrollbar-track { background: transparent; }
  #transcript::-webkit-scrollbar-thumb { background: #3d3d5c; border-radius: 3px; }
</style>
</head>
<body>

<div class="header">
  <h1>🎙️ Thai Voice → English Text</h1>
  <p>AI Realtime Translation · Gemini Live API</p>
</div>

<div class="controls">
  <button id="btnStart" onclick="startSession()">▶ Start</button>
  <button id="btnStop" onclick="stopSession()" disabled>⏹ Stop</button>
  <button id="btnClear" onclick="clearTranscript()">🗑️ Clear</button>
</div>

<div class="status idle" id="status">⚪ Idle — Click Start to begin</div>

<div class="transcript-container">
  <div class="transcript-label">📝 Translation Output</div>
  <div id="transcript">
    <div style="color:#636e72; text-align:center; padding:3rem;">
      🎙️ พูดภาษาไทย แล้วจะแสดงคำแปลที่นี่...<br>
      <small>Text will stream word-by-word in real-time</small>
    </div>
  </div>
</div>

<div class="footer">
  Built with FastAPI + Gemini Live API │ 🎙️ Thai Voice → 🇬🇧 English Text │ PoC
</div>

<script>
let ws = null;
let currentTurnDiv = null;
let currentThLine = null;
let currentEnLine = null;
let turnCount = 0;

function getTimestamp() {
  const now = new Date();
  return now.toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function scrollToBottom() {
  const box = document.getElementById('transcript');
  box.scrollTop = box.scrollHeight;
}

function setStatus(text, cls) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status ' + cls;
}

function ensureTurnBlock() {
  if (!currentTurnDiv) {
    // Remove placeholder if first turn
    if (turnCount === 0) {
      document.getElementById('transcript').innerHTML = '';
    } else {
      // Add divider between turns
      const hr = document.createElement('hr');
      hr.className = 'divider';
      document.getElementById('transcript').appendChild(hr);
    }
    turnCount++;
    currentTurnDiv = document.createElement('div');
    currentTurnDiv.className = 'turn-block';
    document.getElementById('transcript').appendChild(currentTurnDiv);

    // Create TH line
    currentThLine = document.createElement('div');
    currentThLine.className = 'th-line active';
    currentThLine.innerHTML = '<span style="opacity:0.5">🇹🇭 ' + getTimestamp() + ' │ </span>';
    currentTurnDiv.appendChild(currentThLine);

    // Create EN line
    currentEnLine = document.createElement('div');
    currentEnLine.className = 'en-line active';
    currentEnLine.innerHTML = '<span style="opacity:0.5">🇬🇧 ' + getTimestamp() + ' │ </span>';
    currentTurnDiv.appendChild(currentEnLine);
  }
}

function startSession() {
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnStop').disabled = false;
  setStatus('🔗 Connecting...', 'connected');

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(protocol + '//' + location.host + '/ws');

  ws.onopen = () => {
    ws.send(JSON.stringify({action: 'start'}));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'status') {
      if (msg.value === 'connected') {
        setStatus('🔴 Recording & Translating...', 'recording');
      } else if (msg.value === 'disconnected' || msg.value === 'stopped') {
        setStatus('⚪ Stopped', 'idle');
        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnStop').disabled = true;
        // Remove active cursors
        document.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
      }
    }

    else if (msg.type === 'thai') {
      ensureTurnBlock();
      // Append Thai text chunk
      currentThLine.insertAdjacentText('beforeend', msg.text);
      scrollToBottom();
    }

    else if (msg.type === 'english') {
      ensureTurnBlock();
      // Append English text chunk — this comes word-by-word!
      currentEnLine.insertAdjacentText('beforeend', msg.text);
      scrollToBottom();
    }

    else if (msg.type === 'turn_complete') {
      // Remove blinking cursor from completed lines
      if (currentThLine) currentThLine.classList.remove('active');
      if (currentEnLine) currentEnLine.classList.remove('active');
      // Reset for next turn
      currentTurnDiv = null;
      currentThLine = null;
      currentEnLine = null;
      scrollToBottom();
    }

    else if (msg.type === 'error') {
      setStatus('❌ Error: ' + msg.value, 'idle');
    }
  };

  ws.onclose = () => {
    setStatus('⚪ Disconnected', 'idle');
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnStop').disabled = true;
  };
}

function stopSession() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({action: 'stop'}));
  }
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnStop').disabled = true;
}

function clearTranscript() {
  document.getElementById('transcript').innerHTML =
    '<div style="color:#636e72; text-align:center; padding:3rem;">🎙️ พูดภาษาไทย แล้วจะแสดงคำแปลที่นี่...</div>';
  currentTurnDiv = null;
  currentThLine = null;
  currentEnLine = null;
  turnCount = 0;
}
</script>

</body>
</html>
"""


@app.get("/")
async def get_page():
    return HTMLResponse(HTML_PAGE)


if __name__ == "__main__":
    print("=" * 60)
    print("🎙️  Thai Voice → English Text | REAL-TIME STREAMING")
    print("=" * 60)
    print()
    print("🌐 Open http://localhost:8000 in your browser")
    print("💡 Click Start → speak Thai → watch text flow!")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
