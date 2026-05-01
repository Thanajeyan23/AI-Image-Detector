# AI Image Detector Chatbot 🤖📷

A local web application that detects whether an image is **AI-generated or a real photograph**, then lets you chat with an AI assistant about the result.

Built with **PyTorch (ResNet-18)** for binary image classification and **Ollama (llama3.2-vision)** for natural language explanations.

---

## Features

- 🖼️ **Drag-and-drop image upload** — supports PNG, JPG, JPEG, GIF, BMP, WEBP
- 🔍 **AI/Real detection** — custom-trained ResNet-18 with confidence score
- 💬 **Multi-turn chat** — ask follow-up questions about any image
- ⚡ **GPU-accelerated** — runs on CUDA (RTX 3050+) or CPU fallback
- 🔒 **100% local** — no data is sent to external servers

---

## How It Works

```
User uploads image
       ↓
ResNet-18 detector → AI Generated / Real + confidence %
       ↓
User sends a message
       ↓
llama3.2-vision (via Ollama) explains the result visually
       ↓
Multi-turn conversation continues
```

---

## Project Structure

```
CHATBOT/
├── app.py               # Flask server (entry point)
├── requirements.txt     # Python dependencies
│
├── src/
│   ├── model.py         # ResNet-18 model definition (transfer learning)
│   ├── dataset.py       # Data loaders & image augmentation
│   ├── train.py         # Training script
│   ├── predict.py       # Single-image inference
│   └── chat.py          # Ollama multi-turn chat session
│
├── templates/
│   └── index.html       # Chat UI (single-page app)
│
├── data/                # ← NOT in repo (too large: 120K images)
│   ├── train/fake/
│   ├── train/real/
│   ├── val/fake/
│   └── val/real/
│
├── models/              # ← NOT in repo (binary weights)
│   └── detector.pth
│
└── uploads/             # ← NOT in repo (user uploads, temporary)
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU recommended (CPU fallback supported)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (CUDA 12.1):** PyTorch with CUDA is listed in requirements.txt.
> **CPU users:** Replace the torch lines in requirements.txt with the standard CPU versions from [pytorch.org](https://pytorch.org).

### 4. Pull the Ollama vision model

```bash
ollama pull llama3.2-vision
```

### 5. Prepare training data

Organize your dataset like this:

```
data/
├── train/
│   ├── fake/   ← AI-generated images
│   └── real/   ← Real photographs
└── val/
    ├── fake/
    └── real/
```

### 6. Train the model (skip if you have `detector.pth`)

```bash
python src/train.py
```

This will save the best model to `models/detector.pth`.

---

## Running the App

```bash
# Terminal 1 — Start Ollama
ollama serve

# Terminal 2 — Start Flask
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | ResNet-18 (ImageNet pre-trained) |
| Fine-tuned layers | `layer4` + `fc` |
| Output classes | `fake` (AI-generated), `real` (photograph) |
| Input size | 224 × 224 RGB |
| Training images | 100,000 (50K fake, 50K real) |
| Validation images | 20,000 (10K fake, 10K real) |
| Optimizer | Adam (lr=0.001) |
| Early stopping | Patience = 3 epochs |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Flask 3.x |
| Deep learning | PyTorch 2.5 + torchvision |
| LLM / Vision | Ollama + llama3.2-vision |
| Image processing | Pillow |
| Frontend | HTML + CSS + Vanilla JS |

---

## License

MIT License — free to use and modify.
