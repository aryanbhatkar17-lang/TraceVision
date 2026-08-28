<div align="center">

<img src="public/tracevision-icon.svg" alt="TraceVision Logo" width="72" height="72" />

# TraceVision

**AI-Powered CCTV Video Summarization & Forensic Search**

*Smart India Hackathon 2026 — Internal Police & Forensic Investigation Portal*

[![Next.js](https://img.shields.io/badge/Next.js-16.3-black?logo=next.js&logoColor=white)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Gemini](https://img.shields.io/badge/Gemini_API-3.5_Flash_Lite-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

**TraceVision** is an AI-powered forensic investigation portal that allows law enforcement and security analysts to locate specific visual events within hours of CCTV footage using plain-language natural language queries — no manual scrubbing required.

Built specifically for challenging real-world surveillance conditions including low-light environments, low-resolution cameras, and heavy multi-stream recordings, TraceVision combines a **Next.js** investigation dashboard with a **Python FastAPI** video intelligence backend, powered by **Google Gemini** multimodal AI and **Zero-DCE** deep learning enhancement.

```
Upload CCTV footage → Describe what you're looking for → Review matched timestamps
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Natural Language Search** | Query footage with plain text — *"red motorbike near gate"*, *"person in dark hoodie"* |
| 🌑 **Low-Light Enhancement** | Zero-DCE deep curve estimation network enhances dark frames before analysis |
| 🎬 **Smart Clip Extraction** | FFmpeg-accelerated zero-VRAM stream-copy cuts matching segments from source video |
| 🧠 **Gemini Visual Analysis** | Multimodal AI validates clip relevance with semantic scene understanding |
| 📊 **Temporal Smoothing** | Eliminates false positives through confidence-weighted temporal scoring |
| ⚡ **Streaming SSE Pipeline** | Real-time progress updates streamed from backend to dashboard via Server-Sent Events |
| 🔒 **Authorized Access Only** | Designed for closed internal police and forensic investigation networks |
| 🖥️ **Hardware-Aware** | CPU/RAM monitoring with automatic VRAM ceiling enforcement for constrained deployments |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Nginx (port 80)                  │
│         Reverse proxy · 500 MB upload limit · SSE       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────────────────┐
│  Next.js 16.3   │    │     FastAPI (Python 3.10+)        │
│  Investigation  │    │     Video Intelligence Backend    │
│  Dashboard      │◄──►│                                  │
│  (port 3000)    │SSE │  ┌─────────┐  ┌───────────────┐  │
│                 │    │  │ OpenCV  │  │  Zero-DCE     │  │
│  React 19       │    │  │ Frame   │  │  Low-Light    │  │
│  Tailwind CSS 4 │    │  │ Extract │  │  Enhancement  │  │
│  TypeScript     │    │  └────┬────┘  └───────┬───────┘  │
└─────────────────┘    │       │               │           │
                       │  ┌────▼───────────────▼───────┐  │
                       │  │   Temporal Smoother +       │  │
                       │  │   Clip Engine (FFmpeg)      │  │
                       │  └──────────────┬──────────────┘  │
                       │                 │                  │
                       │  ┌──────────────▼──────────────┐  │
                       │  │   Google Gemini Multimodal   │  │
                       │  │   Visual Validation          │  │
                       │  └─────────────────────────────┘  │
                       └──────────────────────────────────┘
```

---

## Tech Stack

### Frontend
- **[Next.js 16.3](https://nextjs.org)** — App Router, SSE client, API gateway
- **[React 19](https://react.dev)** — UI rendering
- **[Tailwind CSS 4](https://tailwindcss.com)** — Utility-first styling with glassmorphism design
- **[TypeScript 5](https://typescriptlang.org)** — End-to-end type safety
- **[Lucide React](https://lucide.dev)** — Icon system

### Backend
- **[FastAPI](https://fastapi.tiangolo.com)** — Async REST API with streaming support
- **[OpenCV](https://opencv.org)** (`opencv-python-headless`) — Frame extraction & motion scoring
- **[FFmpeg](https://ffmpeg.org)** — Zero-VRAM stream-copy clip cutting
- **[PyTorch](https://pytorch.org)** — Zero-DCE enhancement model inference
- **[Google Gemini API](https://ai.google.dev)** (`google-genai`) — Multimodal scene validation
- **[NumPy](https://numpy.org)** — Numerical frame analysis
- **[psutil](https://psutil.readthedocs.io)** — Hardware resource monitoring

### Infrastructure
- **[Nginx](https://nginx.org)** — Reverse proxy, 500 MB upload body, SSE pass-through
- **[Uvicorn](https://www.uvicorn.org)** — ASGI server for FastAPI

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Node.js | ≥ 18.x | For the Next.js dashboard |
| Python | ≥ 3.10 | For the FastAPI backend |
| FFmpeg | ≥ 4.x | Must be on system `PATH` |
| Google Gemini API Key | — | [Get one here](https://ai.google.dev/gemini-api/docs/api-key) |

**Install FFmpeg:**
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/SIH-2026.git
cd SIH-2026
```

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp env.example .env.local
```

Open `.env.local` and fill in your credentials:

```env
# Required: Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# CORS origins for the Python backend (comma-separated)
CORS_ORIGINS=http://localhost:3000

# Node environment
NODE_ENV=development
```

Also create `backend/.env` with the same `GEMINI_API_KEY` (the backend reads from its own directory):

```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > backend/.env
```

---

### 3. Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at **[http://localhost:3000](http://localhost:3000)**.

---

### 4. Backend Setup

```bash
cd backend

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python run.py
# or directly with uvicorn:
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at **[http://localhost:8000](http://localhost:8000)**.

---

### 5. (Optional) Nginx Reverse Proxy

For production-like setups that consolidate the frontend and backend behind a single port:

```bash
# Copy the provided config
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo nginx -t && sudo nginx -s reload
```

Access everything through **[http://localhost:80](http://localhost:80)**.

---

## Investigation Workflow

```
1. Load Video         →  Upload MP4 / AVI / MKV footage (up to 500 MB)
2. Type What You Seek →  Enter a plain-language query: "red motorbike", "person in dark hoodie"
3. Review Evidence    →  Click matched timestamps · Play key moments · Inspect time markers
```

The backend pipeline automatically:

1. **Extracts frames** at adaptive FPS (2fps for clips ≤30s, 1fps for longer footage)
2. **Enhances low-light frames** using Zero-DCE deep curve estimation
3. **Scores motion** and filters static/irrelevant segments
4. **Applies temporal smoothing** to suppress false positives
5. **Validates top-K clips** with Gemini multimodal visual AI
6. **Cuts matching segments** via FFmpeg stream-copy (zero VRAM re-encoding)
7. **Streams results** back to the dashboard in real time via SSE

---

## Project Structure

```
SIH-2026/
├── src/                        # Next.js frontend
│   ├── app/
│   │   ├── page.tsx            # Landing page
│   │   ├── dashboard/          # Investigation dashboard
│   │   └── api/                # Next.js API routes (proxy layer)
│   ├── components/             # Reusable React components
│   ├── lib/                    # Client utilities
│   └── types/                  # TypeScript type definitions
├── backend/                    # Python FastAPI backend
│   ├── server.py               # FastAPI app & Gemini integration
│   ├── pipeline.py             # End-to-end video analysis pipeline
│   ├── clip_engine.py          # Frame scoring & clip selection
│   ├── clip_cutter.py          # FFmpeg / OpenCV clip extraction
│   ├── enhance.py              # Zero-DCE low-light enhancement
│   ├── smoothing.py            # Temporal confidence smoother
│   ├── hardware_monitor.py     # CPU / RAM / VRAM monitoring
│   ├── model.py                # Zero-DCE PyTorch model definition
│   ├── run.py                  # Server entry point
│   └── requirements.txt        # Python dependencies
├── public/                     # Static assets
├── nginx.conf                  # Nginx reverse proxy configuration
├── env.example                 # Environment variable template
├── next.config.ts              # Next.js configuration
└── package.json                # Node.js dependencies & scripts
```

---

## Available Scripts

### Frontend

| Command | Description |
|---|---|
| `npm run dev` | Start Next.js development server with hot reload |
| `npm run build` | Create optimized production build |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |

### Backend

| Command | Description |
|---|---|
| `python run.py` | Start FastAPI server (Uvicorn) |
| `pytest backend/` | Run all backend tests |
| `pytest backend/test_api.py` | Run API integration tests |
| `pytest backend/test_pipeline.py` | Run pipeline unit tests |
| `pytest backend/test_e2e.py` | Run end-to-end tests |

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | — | Google Gemini API key for multimodal analysis |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated allowed CORS origins |
| `NODE_ENV` | No | `development` | Node.js environment mode |
| `ZERO_DCE_WEIGHTS` | No | `Epoch99.pth` | Path to Zero-DCE model weights file |
| `ZERO_DCE_DEVICE` | No | Auto-detect | Force inference device: `cpu`, `cuda`, `cuda:0`, `mps` |

---

## Deployment Notes

- **Upload limit**: Nginx is configured for **500 MB** request bodies (`client_max_body_size 500M`) to handle multi-minute HD footage.
- **Timeouts**: All proxy and client body timeouts are set to **300s** to prevent gateway errors on large uploads and long analysis jobs.
- **SSE Streaming**: `proxy_buffering off` is required for real-time result streaming to work through Nginx.
- **FFmpeg fallback**: If FFmpeg is unavailable on the host, `clip_cutter.py` automatically falls back to OpenCV frame-by-frame re-encoding (slower, but functional on all platforms).
- **GPU acceleration**: Install `pynvml` and uncomment it in `requirements.txt` to enable VRAM monitoring on GPU hosts.

---

## Contributing

This project was built for **Smart India Hackathon 2026**. Contributions, suggestions, and issue reports are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ for **Smart India Hackathon 2026**

*Authorized for internal police & forensic investigation use only.*

</div>
