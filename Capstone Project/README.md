# AI-Powered Transcript-to-Notes System

An intelligent lecture-processing platform that converts YouTube lecture content into structured, study-oriented notes using speech recognition and Large Language Model (LLM) pipelines.

The system is designed to improve learning efficiency by transforming lengthy or fast-paced educational videos into concise, readable, and organized academic material.

---

## Overview

This application automates the workflow of:

1. Extracting lecture audio from YouTube videos
2. Transcribing spoken content into text
3. Processing transcripts using AI-based NLP pipelines
4. Generating structured notes and summaries
5. Enabling interactive understanding of lecture material

The platform is intended for educational assistance, academic revision, and productivity enhancement.

---

## Key Features

- Automated lecture transcription
- AI-generated structured notes
- Concise and detailed summarization modes
- Topic and key-point extraction
- Interactive lecture Q&A support
- Downloadable study material
- Streamlit-based web interface
- Modular and extensible architecture

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend Interface | Streamlit |
| Speech Recognition | Whisper / STT Models |
| NLP Processing | LLM Pipelines |
| API Integration | OpenAI-Compatible HuggingFace APIs |
| Environment Management | Python Virtual Environment |

---

## System Requirements

Before running the application, ensure the following are installed:

- Python 3.10 or above
- Git
- An active OpenAI-compatible API key

### Official Downloads

- Python: https://www.python.org/
- Git: https://git-scm.com/

---
## Additional Requirement

This project requires FFmpeg for audio extraction and transcription processing.

### Install FFmpeg

- Windows: https://www.gyan.dev/ffmpeg/builds/
- macOS:
  ```bash
  brew install ffmpeg
  ```
- Linux:
  ```bash
  sudo apt install ffmpeg
  ```

# Installation Guide

## 1. Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/your-username/Lecture-Note-AI.git
cd Lecture-Note-AI
```

---

## 2. Create a Virtual Environment

Using a virtual environment is strongly recommended to isolate project dependencies.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Project Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

# API Key Configuration

The application requires an API key to access foundational AI and NLP models.

Generate an API key from your preferred OpenAI-compatible provider dashboard.

---

## Configure Environment Variable

### Windows CMD

```bash
set OPENAI_API_KEY=your_api_key_here
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

### macOS / Linux

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

# Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Once executed, the application will automatically launch in your default web browser.

---

# Project Architecture

```text
Lecture-Note-AI/
│
├── app.py
├── requirements.txt
└── README.md
```

The current implementation follows a lightweight Streamlit-based architecture using Python.  
The project is designed to be modular and scalable, with future separation planned for:
- NLP pipelines
- utility functions
- data processing modules
- transcript management
- model integrations
```

---

# Workflow Pipeline

1. YouTube lecture input
2. Audio extraction and preprocessing
3. Speech-to-text transcription
4. NLP and summarization processing
5. Structured note generation
6. User interaction and export

---

# Applications

This platform can be used for:

- Academic lecture summarization
- Revision and exam preparation
- Long-form educational content analysis
- Accessibility support for learners
- Productivity enhancement for students

---

# Future Enhancements

- Multi-language transcription  chat support
- Timestamp-linked notes
- Real-time lecture processing
- Semantic lecture search

---

# Disclaimer

This project is developed for educational and research purposes only.

---

# License

This project is licensed under the MIT License.
