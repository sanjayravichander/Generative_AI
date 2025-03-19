import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Text to Math Solver and Data Assistant", page_icon="🤖")
st.title("🤖 Text to Math Solver and Data Assistant")

# Sidebar for API key
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It", temperature=0.52)

# Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=3)
wiki_tool = Tool(
    name="Wikipedia",
    description="A tool for searching solutions for Math problems",
    func=wiki_wrapper.run
)

# Math Calculation Tool
llm_math_tool = LLMMathChain(llm=llm)
calculator = Tool(
    name="Calculator",
    description="A tool for performing math calculations",
    func=llm_math_tool.run
)

# Reasoning Tool
prompt = """ 
You're an expert in solving math problems. Logically arrive at the solution,
provide a detailed step-by-step explanation, and display it in bullet points.

Question: {question}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning",
    description="A tool for reasoning and solving math problems",
    func=chain.run
)

# Initialize Agent
agent = initialize_agent(
    tools=[wiki_tool, calculator, reasoning_tool],
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)

# Store Chat Messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "Hi, I'm a chatbot who can solve math problems!"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
question = st.chat_input(placeholder="Ask me anything")

# Process Response
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(question, callbacks=[st_cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
