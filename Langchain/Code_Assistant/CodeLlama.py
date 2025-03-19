import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="AI Code Assistant", page_icon="💻")
st.title("💻 AI Code Assistant with Groq 🤖")

# Sidebar for API key
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
if not groq_api_key:
    st.info("Please enter your **Groq API Key** to continue.")
    st.stop()

# Initialize Groq LLM for Coding Tasks
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Qwen-2.5-Coder-32b", temperature=0.5)

# Wikipedia Tool (For Technical Searches)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=3)
wiki_tool = Tool(
    name="Wikipedia",
    description="Search Wikipedia for programming-related topics",
    func=wiki_wrapper.run
)

# Code Explanation & Debugging Tool
code_prompt = """
You're an AI Code Assistant specializing in Python, JavaScript, and other programming languages.
You can:
- **Explain** complex code in a simple way
- **Fix bugs** and optimize code
- **Generate new code** based on requirements
- **Suggest best practices** for software development

Here is the user's request:
{code}
Answer:
"""
code_prompt_template = PromptTemplate(input_variables=["code"], template=code_prompt)
code_chain = LLMChain(llm=llm, prompt=code_prompt_template)

code_tool = Tool(
    name="Code Assistant",
    description="Explain, debug, or generate code",
    func=code_chain.run
)

# Initialize AI Agent
agent = initialize_agent(
    tools=[wiki_tool, code_tool],
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)

# Store Chat Messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "Hi, I'm an AI Code Assistant! Ask me anything about programming."}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
query = st.chat_input(placeholder="Ask me about code...")

# Process Response
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.spinner("Thinking..."):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(query, callbacks=[st_cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
