"""
🎙️ AI Realtime Translate: Thai Voice → English Text
Streamlit Web UI — Push-to-Talk & Always Listening
Powered by Gemini Live API
"""
import asyncio
import io
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from gemini_client import run_live_session

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
if not os.getenv("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="🎙️ TH Voice → EN Text",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Thai:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"], .stApp {
        font-family: 'Inter', 'Noto Sans Thai', sans-serif !important;
        background-color: #07070f !important;
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    /* ── Header ─────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2.5rem 2rem 2rem;
        background: #12103a;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(139,92,246,0.25);
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(139,92,246,0.2);
        border: 1px solid rgba(139,92,246,0.4);
        border-radius: 50px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #c4b5fd;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        margin: 0 0 0.5rem;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
        background: linear-gradient(135deg, #fff 0%, #c4b5fd 60%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .main-header p {
        color: rgba(255,255,255,0.5);
        margin: 0;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }

    /* ── Status badge ───────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.55rem 1.3rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .status-recording {
        background: rgba(239,68,68,0.15);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.35);
        animation: pulse-rec 2s ease-in-out infinite;
    }
    @keyframes pulse-rec {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
        50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
    }
    .status-connected {
        background: rgba(34,197,94,0.12);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .status-idle {
        background: rgba(148,163,184,0.08);
        color: #94a3b8;
        border: 1px solid rgba(148,163,184,0.18);
    }

    /* ── Transcript box ─────────────────────────────── */
    .transcript-box {
        background: #0c0c1a;
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 55vh;
        max-height: 72vh;
        overflow-y: auto;
        border: 1px solid rgba(139,92,246,0.12);
        scroll-behavior: smooth;
    }
    .transcript-box::-webkit-scrollbar { width: 5px; }
    .transcript-box::-webkit-scrollbar-track { background: transparent; }
    .transcript-box::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.3); border-radius: 4px; }

    /* ── Thai row ───────────────────────────────────── */
    .thai-text {
        display: flex;
        align-items: baseline;
        gap: 0.7rem;
        color: #93c5fd;
        font-size: 1.5rem;
        font-weight: 500;
        line-height: 1.75;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.3rem;
        border-radius: 12px;
        background: rgba(96,165,250,0.05);
        border-left: 3px solid rgba(96,165,250,0.5);
        word-break: break-word;
    }
    .thai-text .flag { flex-shrink: 0; }
    .thai-text .content { flex: 1; min-width: 0; }
    .thai-text .ts { font-size: 0.75rem; color: rgba(147,197,253,0.4); white-space: nowrap; }

    /* ── English row ────────────────────────────────── */
    .english-text {
        display: flex;
        align-items: baseline;
        gap: 0.7rem;
        color: #6ee7b7;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.6;
        padding: 1rem 1.4rem;
        margin-bottom: 1.5rem;
        border-radius: 14px;
        background: rgba(52,211,153,0.06);
        border-left: 4px solid rgba(52,211,153,0.55);
        word-break: break-word;
        letter-spacing: -0.2px;
    }
    .english-text .flag { flex-shrink: 0; }
    .english-text .content { flex: 1; min-width: 0; }
    .english-text .ts { font-size: 0.75rem; color: rgba(110,231,183,0.4); white-space: nowrap; }

    /* ── Empty state ────────────────────────────────── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 40vh;
        gap: 0.8rem;
    }
    .empty-state .icon { font-size: 4rem; }
    .empty-state .msg  { font-size: 1.3rem; font-weight: 600; color: rgba(255,255,255,0.25); }
    .empty-state .sub  { font-size: 0.95rem; color: rgba(255,255,255,0.14); }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.65rem 1.5rem !important;
    }

    /* ── Section heading ─────────────────────────────── */
    .section-heading {
        font-size: 1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.5);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* ── Metrics ─────────────────────────────────────── */
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; color: #a78bfa !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: rgba(255,255,255,0.4) !important; }

    /* ── Caption ─────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] { font-size: 0.85rem !important; color: rgba(255,255,255,0.2) !important; text-align: center; }

    /* ── Hide Streamlit chrome ───────────────────────── */
    #MainMenu { visibility: hidden; }
    header    { visibility: hidden; }
    footer    { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────
def init_state():
    defaults = {
        "transcript_history": [],
        "is_recording": False,
        "mode": "always_listening",
        "translation_count": 0,
        "session_start": None,
        "gemini_connected": False,
        "audio_queue": queue.Queue(),
        "result_queue": queue.Queue(),
        "stop_event": threading.Event(),
        "worker_thread": None,
        "mic_thread": None,
        "recording_event": threading.Event(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Gemini background worker ─────────────────────────────
def gemini_worker(api_key, audio_q, result_q, stop_ev):
    """Background thread: Gemini Live API send/receive."""
    # Convert threading.Event → asyncio.Event inside the loop
    async def _run():
        async_stop = asyncio.Event()

        # Bridge: poll threading stop_ev → set asyncio async_stop
        async def _watch_stop():
            while not stop_ev.is_set():
                await asyncio.sleep(0.1)
            async_stop.set()

        asyncio.create_task(_watch_stop())

        try:
            await run_live_session(
                api_key=api_key,
                audio_in=audio_q,
                result_out=result_q,
                stop_event=async_stop,
            )
        except Exception as e:
            result_q.put({"error": f"Session: {e}"})

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


# ─── Mic background worker ────────────────────────────────
def mic_worker(audio_q, stop_ev, rec_ev, always_on):
    """Background thread: PyAudio mic capture (local only)."""
    try:
        import pyaudio
    except ImportError:
        return  # No pyaudio → use browser audio_input instead

    RATE = 16000
    CHUNK = 800  # 50 ms — smaller chunks for faster VAD response

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
    except Exception:
        p.terminate()
        return  # No mic device available (e.g. Streamlit Cloud)

    try:
        while not stop_ev.is_set():
            if always_on or rec_ev.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_q.put(data)
                except OSError:
                    pass
            else:
                time.sleep(0.05)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ─── Helper functions ──────────────────────────────────────
def _flush(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def start_session():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in .env!")
        return False

    st.session_state.stop_event.clear()
    _flush(st.session_state.audio_queue)
    _flush(st.session_state.result_queue)

    always_on = st.session_state.mode == "always_listening"
    if always_on:
        st.session_state.recording_event.set()
    else:
        st.session_state.recording_event.clear()

    # Gemini thread
    wt = threading.Thread(
        target=gemini_worker,
        args=(api_key, st.session_state.audio_queue,
              st.session_state.result_queue, st.session_state.stop_event),
        daemon=True,
    )
    wt.start()
    st.session_state.worker_thread = wt

    # Mic thread
    mt = threading.Thread(
        target=mic_worker,
        args=(st.session_state.audio_queue, st.session_state.stop_event,
              st.session_state.recording_event, always_on),
        daemon=True,
    )
    mt.start()
    st.session_state.mic_thread = mt
    st.session_state.session_start = datetime.now()
    return True


def stop_session():
    st.session_state.stop_event.set()
    st.session_state.recording_event.clear()
    st.session_state.is_recording = False
    st.session_state.gemini_connected = False
    for t in [st.session_state.worker_thread, st.session_state.mic_thread]:
        if t:
            t.join(timeout=2)
    st.session_state.worker_thread = None
    st.session_state.mic_thread = None


def poll_results():
    out = []
    while not st.session_state.result_queue.empty():
        try:
            out.append(st.session_state.result_queue.get_nowait())
        except queue.Empty:
            break
    return out


# ═══════════════════════════════════════════════════════════
#                         U I
# ═══════════════════════════════════════════════════════════

# ─── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">✦ Gemini Live API · Real-time AI</div>
    <h1>🎙️ Thai Voice → English Text</h1>
    <p>พูดภาษาไทย · รับคำแปลภาษาอังกฤษแบบ Real-time</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = os.getenv("GEMINI_API_KEY", "")
    st.success("✅ API Key loaded") if api_key else st.error("❌ API Key missing")

    st.markdown("---")
    st.markdown("### 🎚️ Mode")
    mode = st.radio(
        "เลือกโหมด:",
        ["always_listening", "push_to_talk"],
        format_func=lambda x: (
            "🎙️ Always Listening (ฟังตลอด)" if x == "always_listening"
            else "🔘 Push-to-Talk (กดเพื่อพูด)"
        ),
        key="mode_radio",
    )
    st.session_state.mode = mode

    st.markdown("---")
    st.markdown("### 📖 วิธีใช้")
    if mode == "always_listening":
        st.info("1. กด **▶ Start**\n2. พูดภาษาไทย\n3. ดูคำแปลด้านล่าง")
    else:
        st.info("1. กด **▶ Start**\n2. กด **🔴 Record** เมื่อจะพูด\n3. กด **⏹ Stop** เมื่อจบ")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("แปลแล้ว", st.session_state.translation_count)
    with c2:
        if st.session_state.session_start:
            s = (datetime.now() - st.session_state.session_start).seconds
            st.metric("เวลา", f"{s // 60}:{s % 60:02d}")
        else:
            st.metric("เวลา", "0:00")


# ─── Connection controls ──────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    if not st.session_state.gemini_connected:
        if st.button("▶ Start Session", type="primary", use_container_width=True):
            with st.spinner("🔗 Connecting to Gemini Live API..."):
                if start_session():
                    ok = False
                    for _ in range(50):
                        for r in poll_results():
                            if r.get("status") == "connected":
                                ok = True
                            if "error" in r:
                                st.error(f"❌ {r['error']}")
                        if ok:
                            break
                        time.sleep(0.1)

                    if ok:
                        st.session_state.gemini_connected = True
                        st.session_state.is_recording = (
                            st.session_state.mode == "always_listening"
                        )
                        st.rerun()
                    else:
                        stop_session()
                        st.error("❌ Connection timeout")
    else:
        if st.button("⏹ Stop Session", type="secondary", use_container_width=True):
            stop_session()
            st.rerun()

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.transcript_history = []
        st.session_state.translation_count = 0
        st.rerun()

# ─── Status ────────────────────────────────────────────────
if st.session_state.gemini_connected and st.session_state.is_recording:
    st.markdown(
        '<div class="status-badge status-recording">'
        '🔴 Recording &amp; Translating...</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.gemini_connected:
    st.markdown(
        '<div class="status-badge status-connected">'
        '🟢 Connected — Ready</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-badge status-idle">'
        '⚪ Idle</div>',
        unsafe_allow_html=True,
    )


# ─── Push-to-Talk buttons ─────────────────────────────────
if st.session_state.gemini_connected and st.session_state.mode == "push_to_talk":
    st.markdown("---")
    pc1, pc2 = st.columns(2)
    with pc1:
        if not st.session_state.is_recording:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                _flush(st.session_state.audio_queue)
                st.session_state.recording_event.set()
                st.session_state.is_recording = True
                st.rerun()
    with pc2:
        if st.session_state.is_recording:
            if st.button("⏹ Stop Recording", type="secondary", use_container_width=True):
                st.session_state.recording_event.clear()
                st.session_state.is_recording = False
                st.rerun()

# ─── Browser Audio Input (Streamlit Cloud fallback) ───────
# Use st.audio_input when pyaudio is unavailable (no mic on server)
if st.session_state.gemini_connected:
    _has_pyaudio = False
    try:
        import pyaudio
        p_test = pyaudio.PyAudio()
        try:
            p_test.get_default_input_device_info()
            _has_pyaudio = True
        except Exception:
            pass
        p_test.terminate()
    except Exception:
        pass

    if not _has_pyaudio:
        st.markdown("---")
        st.markdown("### 🎤 Browser Microphone")
        audio_bytes = st.audio_input("กดปุ่มไมค์เพื่อพูด แล้วกดหยุดเมื่อจบ")
        if audio_bytes is not None:
            # Convert WAV bytes → raw PCM int16 at 16kHz and push to queue
            import wave, struct
            raw = audio_bytes.read()
            try:
                with wave.open(io.BytesIO(raw)) as wf:
                    n_ch = wf.getnchannels()
                    src_rate = wf.getframerate()
                    pcm = wf.readframes(wf.getnframes())
                # Convert to numpy int16
                samples = np.frombuffer(pcm, dtype=np.int16)
                # Mix down to mono if stereo
                if n_ch > 1:
                    samples = samples.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
                # Resample to 16kHz if needed (simple linear interp)
                if src_rate != 16000:
                    duration = len(samples) / src_rate
                    target_len = int(duration * 16000)
                    xs = np.linspace(0, len(samples) - 1, target_len)
                    samples = np.interp(xs, np.arange(len(samples)), samples).astype(np.int16)
                # Push in 50ms chunks (800 samples @ 16kHz)
                CHUNK = 800
                for i in range(0, len(samples), CHUNK):
                    chunk = samples[i:i + CHUNK]
                    if len(chunk) < CHUNK:
                        chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
                    st.session_state.audio_queue.put(chunk.tobytes())
            except Exception as e:
                st.warning(f"Audio processing error: {e}")


# ─── Transcript area ──────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">📝 Translation Output</div>', unsafe_allow_html=True)

# Poll new results from Gemini
for r in poll_results():
    ts = datetime.now().strftime("%H:%M:%S")

    if "thai" in r:
        hist = st.session_state.transcript_history
        if hist and hist[-1]["type"] == "thai" and not hist[-1].get("done"):
            # append chunk — keep original timestamp (don't overwrite)
            hist[-1]["text"] += r["thai"]
        else:
            hist.append({"type": "thai", "text": r["thai"], "time": ts, "done": False})

    if "english" in r:
        hist = st.session_state.transcript_history
        if hist and hist[-1]["type"] == "english" and not hist[-1].get("done"):
            # append chunk — keep original timestamp (don't overwrite)
            hist[-1]["text"] += r["english"]
        else:
            hist.append({"type": "english", "text": r["english"], "time": ts, "done": False})
        st.session_state.translation_count = sum(1 for e in hist if e["type"] == "english")

    if "english_text" in r:
        hist = st.session_state.transcript_history
        hist.append({"type": "english", "text": r["english_text"], "time": ts, "done": False})
        st.session_state.translation_count = sum(1 for e in hist if e["type"] == "english")

    if r.get("turn_complete"):
        for e in st.session_state.transcript_history:
            e["done"] = True

    if "error" in r:
        st.error(f"❌ {r['error']}")

# Render transcript
if st.session_state.transcript_history:
    parts = []
    for e in st.session_state.transcript_history[-100:]:
        if e["type"] == "thai":
            parts.append(
                f'<div class="thai-text">'
                f'<span class="flag">🇹🇭</span>'
                f'<span class="content">{e["text"]}</span>'
                f'<span class="ts">{e["time"]}</span>'
                f'</div>'
            )
        else:
            parts.append(
                f'<div class="english-text">'
                f'<span class="flag">🇬🇧</span>'
                f'<span class="content">{e["text"]}</span>'
                f'<span class="ts">{e["time"]}</span>'
                f'</div>'
            )
    html = "\n".join(parts)
else:
    html = (
        '<div class="empty-state">'
        '<div class="icon">🎙️</div>'
        '<div class="msg">พูดภาษาไทย แล้วจะแสดงคำแปลที่นี่</div>'
        '<div class="sub">กด Start Session แล้วเริ่มพูดได้เลย</div>'
        '</div>'
    )

st.markdown(f'<div class="transcript-box" id="tbox">{html}</div>', unsafe_allow_html=True)

# Auto-scroll to bottom — works both locally and on Streamlit Cloud
st.markdown(
    """<script>
    (function(){
        function scrollToBottom(){
            // Try in current document first
            var t = document.getElementById("tbox");
            if(t){ t.scrollTop = t.scrollHeight; return; }
            // Streamlit renders inside an iframe — traverse up to parent
            try {
                var frames = window.parent.document.querySelectorAll("iframe");
                frames.forEach(function(f){
                    try {
                        var el = f.contentDocument.getElementById("tbox");
                        if(el){ el.scrollTop = el.scrollHeight; }
                    } catch(e){}
                });
            } catch(e){}
        }
        setTimeout(scrollToBottom, 50);
        setTimeout(scrollToBottom, 200);
        setTimeout(scrollToBottom, 500);
    })();
    </script>""",
    unsafe_allow_html=True,
)


# ─── Auto-refresh while active ────────────────────────────
if st.session_state.gemini_connected:
    time.sleep(0.15)
    st.rerun()


# ─── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption("✦ Built with Streamlit + Gemini Live API  ·  🎙️ Thai Voice → 🇬🇧 English Text  ·  PoC")
