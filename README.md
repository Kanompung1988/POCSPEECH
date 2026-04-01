# 🎙️ AI Realtime Translate: Thai Voice → English Text

Real-time Thai speech to English text translation powered by **Gemini Live API**.

## Features

| Mode | Description |
|------|-------------|
| 🎙️ **Always Listening** | ฟังตลอดเวลา แปลอัตโนมัติ |
| 🔘 **Push-to-Talk** | กดปุ่มค้างเพื่อพูด ปล่อยเพื่อหยุด |

## Quick Start

### 1. Setup
```bash
# .env
GEMINI_API_KEY=your_api_key_here

# Install
pip install -r requirements.txt
```

### 2. Run

**Streamlit Web UI:**
```bash
streamlit run app.py
```

**Terminal mode:**
```bash
python app_terminal.py                  # Always Listening
python app_terminal.py --push-to-talk   # Push-to-Talk
```
