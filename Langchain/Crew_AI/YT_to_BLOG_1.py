import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from datetime import datetime
import os
import googleapiclient.discovery
from googleapiclient.discovery import build

# YouTube API Key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Groq AI API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def get_video_id(url):
    """Extracts video ID from YouTube URL."""
    if "youtube.com/watch" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None

def get_channel_videos(channel_identifier):
    """Fetches the latest 5 videos from a YouTube channel."""
    if "youtube.com/channel/" in channel_identifier:
        channel_id = channel_identifier.split("youtube.com/channel/")[-1].split("?")[0]
    elif "youtube.com/c/" in channel_identifier:
        channel_username = channel_identifier.split("youtube.com/c/")[-1].split("?")[0]
        channel_response = youtube.search().list(q=channel_username, type="channel", part="id", maxResults=1).execute()
        channel_id = channel_response["items"][0]["id"]["channelId"] if channel_response["items"] else None
    else:
        search_response = youtube.search().list(q=channel_identifier, type="channel", part="id", maxResults=1).execute()
        channel_id = search_response["items"][0]["id"]["channelId"] if search_response["items"] else None

    if not channel_id:
        return None

    videos_response = youtube.search().list(
        channelId=channel_id,
        type="video",
        part="id,snippet",
        order="date",
        maxResults=5
    ).execute()

    return [{"title": item["snippet"]["title"], "videoId": item["id"]["videoId"]} for item in videos_response.get("items", [])]

def get_transcript(video_id):
    """Fetches transcript of a given YouTube video ID."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"🚨 Error fetching transcript: {str(e)}")
        return None

def generate_blog_content(transcript):
    """Generates a blog post from a transcript using Groq AI."""
    prompt = f"""Convert this video transcript into a professional blog post:
    - Create catchy SEO-friendly title
    - Structure with introduction, sections, and conclusion
    - Use proper HTML formatting with headings
    - Include bullet points and examples
    - Add relevant emojis in headings
    - Keep the conversational tone
    Transcript: {transcript[:15000]}"""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Professional content writer"},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"🚨 API Error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="YouTube to Blog Converter", page_icon="🎥", layout="wide")
    
    # Header Section
    st.markdown("""
        <h1 style="color: #FFD700;">🎥 YouTube to Blog Converter</h1>
        <p style="color: #FFA500;">Transform any YouTube video into a beautiful blog post ✨</p>
        <hr>
    """, unsafe_allow_html=True)

    url = st.text_input("Paste YouTube Video URL or Channel Name", placeholder="Enter YouTube URL or Channel Name...", key="url_input")

    video_id = None
    selected_video = None
    video_options = []

    if url:
        if "youtube.com/watch" in url or "youtu.be" in url:
            video_id = get_video_id(url)
        else:
            st.write("🔍 Fetching latest videos from channel...")
            video_options = get_channel_videos(url)
            if video_options:
                selected_video = st.selectbox("🎬 Select a video:", video_options, format_func=lambda x: x["title"])
                if selected_video:
                    video_id = selected_video["videoId"]

        if video_id:
            with st.status("🚀 Processing your video...", expanded=True) as status:
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=80)

                with cols[1]:
                    st.write("**Step 1:** Extracting Video ID... ✅")
                    st.write("**Step 2:** Fetching Transcript... ⏳")
                    
                    transcript = get_transcript(video_id)
                    
                    if transcript:
                        st.write("**Step 3:** Generating Blog Content... ⏳")
                        blog_content = generate_blog_content(transcript)
                        
                        if blog_content:
                            status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                            st.balloons()

                            # Result Section
                            with st.container():
                                st.subheader("🎉 Your Generated Blog Post")
                                st.markdown("---")
                                st.markdown(blog_content, unsafe_allow_html=True)

                                # Download Section
                                st.download_button(
                                    label="📥 Download Blog",
                                    data=f"<html><body>{blog_content}</body></html>",
                                    file_name=f"blog_{datetime.now().strftime('%Y%m%d')}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                        else:
                            st.error("Failed to generate content")
                    else:
                        st.error("Could not retrieve transcript")

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1055/1055666.png", width=80)
        st.markdown("### 📍 How It Works")
        st.markdown("""
            1. Paste YouTube Video URL or Channel Name
            2. If a **channel name** is entered, choose a video from the list
            3. AI extracts the transcript & generates a blog
            4. Download or copy the result
        """)
        st.markdown("---")
        st.markdown("**Supported Features:**")
        st.markdown("""
            - 🎥 Video to text conversion
            - ✍️ AI-powered writing
            - 📱 Mobile-friendly
            - 🎨 Beautiful formatting
        """)
        st.markdown("---")
        st.markdown("*Made with ❤️ by Your Name*")

if __name__ == "__main__":
    main()
