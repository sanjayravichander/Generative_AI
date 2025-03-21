import os
from dotenv import load_dotenv
from crewai import Agent
from tools import yt_tool
from langchain_groq import ChatGroq
# Load environment variables first
load_dotenv()

# Explicitly set Groq configuration
api_key=os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm=ChatGroq(temperature=0.6, model_name="llama3-70b-8192",groq_api_key=api_key)
blog_researcher = Agent(
    role="Blog Content Researcher",
    goal="Research content from YouTube videos",
    backstory="Expert researcher analyzing YouTube content",
    verbose=True,
    memory=True,
    tools=[yt_tool],
    llm=llm,  # Use 70b model for better results
    max_iter=5  # Add iteration limit for safety
)

blog_writer_agent = Agent(
    role="Blog Writer",
    goal="Create engaging blog posts",
    backstory="Skilled technical writer crafting narratives",
    verbose=True,
    memory=True,
    tools=[yt_tool],
    max_iter=5,
    llm=llm
)