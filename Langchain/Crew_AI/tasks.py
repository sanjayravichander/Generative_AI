from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer_agent

# Creating the Task
blog_researcher_task = Task(
    description=(
        "Identify the video from {content}. "
        "Get detailed information about the video from the channel."
    ),
    expected_output="A comprehensive 3-paragraph long blog post about the {content} of the video.",
    tools=[yt_tool],
    agent=blog_researcher
)

# Creating a Writing Task
blog_writer_task = Task(
    description=(
        "Compose an engaging blog post based on the video {content}. "
        "Ensure the blog post is informative and captures the essence of the video."
    ),
    expected_output="Summarize the info from the YouTube channel video on the content {content} and create content for the blog.",
    tools=[yt_tool],
    agent=blog_writer_agent,
    context=[blog_researcher_task],
    async_execution=False,
    output_file="blog-post.md"
)
