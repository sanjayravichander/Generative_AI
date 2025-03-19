import streamlit as st
import os
import validators
import yt_dlp
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

# ✅ Move this to the very beginning after imports
st.set_page_config(page_title="YouTube & Website Summarizer", page_icon="📹", layout="wide")

st.title("📹 YouTube and Website Summarizer")
st.subheader("Summarize any YouTube video or webpage!")

# Custom CSS for Styling
st.markdown("""
    <style>
        /* Background Image */
        .stApp {
            background-image: url("https://source.unsplash.com/1600x900/?technology,abstract");
            background-size: cover;
            background-position: center;
            color: white;
        }

        /* Title Styling */
        h1 {
            text-align: center;
            font-size: 45px;
            font-weight: bold;
            color: #ffcc00;
            text-shadow: 2px 2px 5px black;
        }

        /* Subheader Styling */
        h2 {
            text-align: center;
            font-size: 25px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 3px black;
        }

        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #1a1a1a !important;
            color: white !important;
        }

        /* Input Box Styling */
        input {
            border-radius: 10px !important;
            padding: 10px !important;
            border: 2px solid #ffcc00 !important;
        }

        /* Button Styling */
        .stButton button {
            background-color: #ffcc00 !important;
            color: black !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            padding: 10px !important;
            transition: all 0.3s ease-in-out;
        }

        .stButton button:hover {
            background-color: #ff9900 !important;
            transform: scale(1.05);
        }

        /* Spinner Animation */
        .stSpinner {
            color: #ffcc00 !important;
        }

        /* Success Box */
        .stAlert {
            background-color: rgba(0, 0, 0, 0.8) !important;
            color: #ffcc00 !important;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API Key
with st.sidebar:
    api_key = st.text_input("🔑 Enter Groq API Key", type="password")

# URL Input with Custom Placeholder
generic_url = st.text_input("🔗 Enter URL here...", label_visibility="collapsed", placeholder="Paste YouTube or webpage link...")

if not api_key:
    st.error("⚠️ Please enter the Groq API Key.")
else:
    os.environ["GROQ_API_KEY"] = api_key  # Set API key

    # Initialize LLM
    llm = ChatGroq(model="Gemma2-9b-It", temperature=0.5)

# Prompt Template
prompt_template = """ 
You're an expert in summarizing content. Provide a concise summary within 500 words:
content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_youtube_transcript(video_url):
    """Tries to extract a YouTube transcript using YouTubeTranscriptApi and falls back to yt-dlp."""
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript_data])
    except (TranscriptsDisabled, NoTranscriptFound):
        # Fallback to yt-dlp if transcript is disabled or not found
        try:
            ydl_opts = {
                'quiet': True,
                'skip_download': True,
                'writesubtitles': True,
                'subtitleslangs': ['en'],
                'writeautomaticsub': True,
                'outtmpl': '%(id)s.%(ext)s'
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                subtitles = info.get("subtitles") or info.get("automatic_captions")
                if subtitles and "en" in subtitles:
                    url = subtitles["en"][0]["url"]
                    response = requests.get(url)
                    return response.text if response.status_code == 200 else "Error: Unable to fetch subtitles."
        except Exception as e:
            return f"Error extracting transcript: {str(e)}"

    return "Error: No transcript available."

def extract_webpage_content(url):
    """Extracts content from a normal webpage using requests and BeautifulSoup."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Error: Unable to fetch webpage, status code {response.status_code}"
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  # Extract all paragraphs
        text = "\n".join([p.get_text() for p in paragraphs if p.get_text()])
        
        return text if text else "Error: No text content found on the webpage."
    except Exception as e:
        return f"Error extracting webpage content: {str(e)}"

if st.button("🔍 Summarize"):
    if not api_key.strip() or not generic_url.strip():
        st.error("⚠️ Please enter both Groq API Key and URL.")
    elif not validators.url(generic_url):
        st.error("⚠️ Please enter a valid URL.")
    else:
        try:
            with st.spinner("🔄 Extracting content..."):
                if "youtube.com" in generic_url:
                    content = extract_youtube_transcript(generic_url)
                else:
                    content = extract_webpage_content(generic_url)

                if "Error" in content:
                    st.error(content)
                else:
                    # ✅ Convert extracted content into a Document object
                    document = Document(page_content=content)

                    # Summarizing
                    with st.spinner("🔄 Summarizing..."):
                        chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                        summary = chain.invoke({"input_documents": [document]})  # ✅ Fixed

                        st.success(summary["output_text"])
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
