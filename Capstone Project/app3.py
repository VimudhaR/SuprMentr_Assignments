import streamlit as st
import yt_dlp
import whisper
import torch
import os
import time
import threading
from huggingface_hub import InferenceClient
from deep_translator import GoogleTranslator

# ---------------------------------------------------
# PAGE CONFIG & CUSTOM CSS
# ---------------------------------------------------
st.set_page_config(page_title="Lecture Note AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main { padding-top: 1rem; }
div.stButton > button:first-child {
    background-color: #fcf4ba;
    color: #333333;
    border: 1px solid #fce883;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6rem 1rem;
}
div.stButton > button:first-child:hover {
    background-color: #fce883;
    border: 1px solid #e5c100;
    color: black;
}
.stDownloadButton button {
    background-color: #fcf4ba;
    color: black;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Smart Lecture Summarizer")
st.caption("AI-powered multilingual lecture transcription, summarization, translation, and interactive Q&A.")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    language_mapping = {
        "Auto Detect": None, "English": "en", "Hindi": "hi", 
        "Tamil": "ta", "Kannada": "kn", "Telugu": "te", 
        "Spanish": "es", "Chinese": "zh", "Korean": "ko"
    }
    selected_lang = st.selectbox("🗣️ Source Language", list(language_mapping.keys()))
    source_language_code = language_mapping[selected_lang]

    translate_notes = st.toggle("🌐 Translate Final Notes to English", value=False)
    model_option = st.selectbox("🤖 Whisper Model", ["base", "medium"], index=0)
    
    st.divider()
    if torch.cuda.is_available():
        st.success(f"⚡ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("🐢 Running on CPU")

# ---------------------------------------------------
# AUTH & CLIENTS
# ---------------------------------------------------
HF_TOKEN = None
try:
    if "HF_TOKEN" in st.secrets:
        HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    pass
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("🚨 HF_TOKEN not found.")
    st.stop()

hf_client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

# Make sure you have 'import time' at the top of your file!

# ---------------------------------------------------
# AUDIO DOWNLOAD (Updated with Unique Filenames)
# ---------------------------------------------------
def download_audio(link):
    # 1. Create a unique ID based on the current time
    unique_id = int(time.time())
    audio_path = f"lecture_audio_{unique_id}.mp3"
    
    # 2. Shorts support
    if "/shorts/" in link:
        video_id = link.split("/shorts/")[1].split("?")[0]
        link = f"https://www.youtube.com/watch?v={video_id}"
        
    # 3. Clean up OLD files so your hard drive doesn't fill up
    for file in os.listdir():
        if file.startswith("lecture_audio"):
            try: 
                os.remove(file)
            except: 
                # If a file is locked by Windows, we just ignore it and move on
                pass

    # 4. Tell yt-dlp to use our new unique filename
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'lecture_audio_{unique_id}', # <--- The Magic Fix
        'extract_flat': False,
        'noplaylist': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'http_headers': {'User-Agent': 'Mozilla/5.0'}
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
        
    return audio_path

# ---------------------------------------------------
# WHISPER TRANSCRIPTION
# ---------------------------------------------------
@st.cache_resource
def load_whisper_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)

def transcribe_audio(audio_file, language_code, model_name):
    result_container = {}
    error_container = {}

    def run_whisper():
        try:
            model = load_whisper_model(model_name)
            whisper_args = {"task": "transcribe", "verbose": False, "fp16": torch.cuda.is_available()}

            if language_code is not None:
                whisper_args["language"] = language_code
            else:
                audio = whisper.load_audio(audio_file)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                _, probs = model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                whisper_args["language"] = detected_lang

            result = model.transcribe(audio_file, **whisper_args)
            result_container["result"] = result
        except Exception as e:
            error_container["error"] = str(e)

    thread = threading.Thread(target=run_whisper)
    thread.start()

    progress_bar = st.progress(0)
    status_text = st.empty()
    elapsed = 0

    while thread.is_alive():
        fake_progress = min(0.90, elapsed / 120)
        progress_bar.progress(fake_progress)
        messages = ["🎧 Processing audio...", "✍️ Transcribing...", "🧠 Analyzing script..."]
        status_text.text(messages[elapsed % len(messages)])
        time.sleep(1)
        elapsed += 1

    thread.join()
    progress_bar.progress(1.0)
    status_text.text("✅ Transcription complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    if "error" in error_container:
        raise Exception(error_container["error"])
    return result_container["result"]

# ---------------------------------------------------
# BULLETPROOF TRANSLATOR (Auto-Chunks to avoid crashes)
# ---------------------------------------------------
def translate_text(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        # Google Translate has a 5k char limit. We slice it to prevent errors on "Lengthy" notes.
        if len(text) < 4500:
            return translator.translate(text)
        else:
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated_chunks = [translator.translate(c) for c in chunks]
            return "".join(translated_chunks)
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text

# ---------------------------------------------------
# DYNAMIC SUMMARIZER
# ---------------------------------------------------
# ---------------------------------------------------
# DYNAMIC SUMMARIZER (Now Handles Translation + Blocks Chinese)
# ---------------------------------------------------
def summarize(text, style, translate_to_english):
    if style == "Concise":
        style_prompt = "Write a very brief, concise summary using short bullet points. Focus ONLY on the absolute main ideas."
    elif style == "Paragraph":
        style_prompt = "Write a flowing, well-structured essay-style summary in paragraphs. STRICT RULE: DO NOT use any bullet points."
    else:
        style_prompt = "Write highly detailed, comprehensive study notes. Use extensive bullet points, bold key terms, and extract EVERY concept mentioned."

    # Force the language anchor to prevent the Chinese bug
    if translate_to_english:
        lang_rule = "CRITICAL: You MUST write the notes entirely in ENGLISH, translating the transcript context flawlessly."
    else:
        lang_rule = "CRITICAL: You MUST write the notes in the EXACT SAME LANGUAGE as the transcript chunk. DO NOT translate to English."

    messages = [
        {
            "role": "system", 
            "content": (
                f"You are an expert teaching assistant. {style_prompt} "
                f"{lang_rule} "
                "WARNING: DO NOT output any Chinese characters unless the transcript is originally in Chinese. "
                "Fix any obvious phonetic typos in the transcript using context."
            )
        },
        {"role": "user", "content": text}
    ]

    for attempt in range(3):
        try:
            response = hf_client.chat_completion(messages=messages, max_tokens=1500, temperature=0.3)
            return response.choices[0].message.content
        except Exception as e:
            if "loading" in str(e).lower() or "503" in str(e).lower():
                time.sleep(10)
                continue
            else: return None
    return None

def summarize_large_text(text, style, translate_to_english):
    chunk_size = 2500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, chunk in enumerate(chunks):
        status_text.text(f"📝 Summarizing chunk {i+1}/{len(chunks)} ({style} style)...")
        # Pass the translation flag directly to the AI
        summary_piece = summarize(chunk, style, translate_to_english)
        if summary_piece is None:
            st.error(f"Error summarizing chunk {i+1}")
            return None
        summaries.append(summary_piece)
        progress_bar.progress((i + 1) / len(chunks))

    progress_bar.empty()
    status_text.text("✅ Notes generated!")
    
    join_char = "\n\n" if style != "Paragraph" else "\n"
    return join_char.join(summaries)


# ---------------------------------------------------
# CHATBOT
# ---------------------------------------------------
def ask_lecture(question, context):
    safe_context = context[:15000]
    messages = [
        {"role": "system", "content": f"You are a helpful study assistant. Answer ONLY from the lecture context.\n\nContext:\n{safe_context}"},
        {"role": "user", "content": question}
    ]
    try:
        response = hf_client.chat_completion(messages=messages, max_tokens=400, temperature=0.5)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------
# UI FLOW
# ---------------------------------------------------
if "transcript" not in st.session_state: st.session_state["transcript"] = ""
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "notes" not in st.session_state: st.session_state["notes"] = ""

yt_url = st.text_input("🔗 Paste Public YouTube Lecture Link")

if st.button("🚀 Step 1: Extract & Transcribe"):
    if yt_url:
        try:
            with st.status("Processing Lecture...", expanded=True) as status:
                st.write("📥 Downloading audio...")
                audio_file = download_audio(yt_url)
                if os.path.getsize(audio_file) < 1000:
                    st.error("Downloaded audio file is empty.")
                    st.stop()
                st.write(f"🧠 Loading Whisper {model_option} model...")
                
                result = transcribe_audio(audio_file, source_language_code, model_option)
                st.session_state["transcript"] = result["text"]
                detected_lang = result.get("language", "unknown")
                
                status.update(label=f"✅ Done! Detected Language: {detected_lang.upper()}", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a YouTube link.")

if st.session_state["transcript"]:
    with st.expander("📄 View Transcript"):
        st.write(st.session_state["transcript"])
    st.divider()

    col1, col2 = st.columns(2)

    # --- REPLACE YOUR COL1 BLOCK WITH THIS ---
    with col1:
        st.subheader("📝 Smart Notes")
        note_style = st.radio("Style:", ["Concise", "Lengthy", "Paragraph"], horizontal=True, label_visibility="collapsed")

        if st.button("Generate Notes", use_container_width=True):
            # We now pass the 'translate_notes' toggle state directly into the summarizer
            final_notes = summarize_large_text(st.session_state["transcript"], note_style, translate_notes)
            
            if final_notes:
                st.session_state["notes"] = final_notes
                st.markdown(final_notes)
                st.download_button(
                    "⬇️ Download Notes", 
                    final_notes, 
                    file_name=f"lecture_notes_{note_style.lower()}.txt", 
                    use_container_width=True
                )

    with col2:
        st.subheader("💬 Chat with Lecture")
        chat_container = st.container(height=400)
        with chat_container:
            if len(st.session_state["chat_history"]) == 0:
                st.info("Ask questions about the lecture.")
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = ask_lecture(prompt, st.session_state["transcript"])
                        st.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})