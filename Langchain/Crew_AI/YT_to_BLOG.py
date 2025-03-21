# YT_to_BLOG.py (modified)
from crewai import Crew, Process
from agents import blog_researcher, blog_writer_agent
from tasks import blog_researcher_task, blog_writer_task
import streamlit as st

def generate_blog(content):
    crew = Crew(
        agents=[blog_researcher, blog_writer_agent],
        tasks=[blog_researcher_task, blog_writer_task],
        process=Process.sequential,
        memory=True,
        cache=True,
        max_rpm=100,
        share_crew=True
    )
    return crew.kickoff(inputs={"content": content})

# Streamlit interface
def main():
    st.title("YouTube Video to Blog Converter")
    content = st.text_input("Enter YouTube video title/content:", 
                          "What to Do, See & Eat in Obihiro, Japan | Three Day Weekend Itinerary & Travel Guide")
    
    if st.button("Generate Blog Post"):
        with st.spinner("Creating blog post..."):
            result = generate_blog(content)
            st.subheader("Generated Blog Post")
            st.markdown(result)
            st.download_button(
                label="Download Blog Post",
                data=result,
                file_name="blog-post.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()