import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQAI_API_KEY"]=os.getenv("GROQAI_API_KEY")
from langchain_community.tools import DuckDuckGoSearchRun,ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler

arxiv_wrapper=ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=200)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=200)
wiki_tool=WikipediaQueryRun(api_wrapper=wiki_wrapper)
search=DuckDuckGoSearchRun(name="search")

st.title("Langchain Agents with Groq LLM")
st.write("This is Simple Q and A from research paper using tools and agents")
api_key=st.text_input("Enter your Groq API Key: ", type="password")
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a chatbot who can search web.How Can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt:=st.chat_input("What is Machine Learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    llm=ChatGroq(temperature=0,api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[arxiv_tool,wiki_tool,search]
    agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,handling_parsing_error=True)
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response=agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
