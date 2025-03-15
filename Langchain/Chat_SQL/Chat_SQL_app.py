import streamlit as st
import os 
from pathlib import Path
from langchain_chroma import Chroma
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_groq import ChatGroq
import sqlite3
from sqlalchemy import create_engine
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_types import AgentType

# Streamlit App Setup
st.set_page_config(page_title="🤖 Chat with your SQL Database")
st.title("🤖 Chat with your SQL Database")

INJECTION_WARNING = """ 
⚠️ WARNING: SQL Agent can be vulnerable to SQL injection attacks. 
Please ensure that the input is sanitized before using the SQL Agent.
"""
st.sidebar.warning(INJECTION_WARNING)

# Database Connection Options
LOCAL_DB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_option = ["Use SQLITE 3 Database - Students.db", "Connect to your SQL Database"]
select_option = st.sidebar.radio(label="Select Database", options=radio_option)

# Handle Database Selection
if radio_option.index(select_option) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Enter MySQL Host")
    mysql_user = st.sidebar.text_input("Enter MySQL User")
    mysql_password = st.sidebar.text_input("Enter MySQL Password", type="password")
    mysql_database = st.sidebar.text_input("Enter MySQL Database")

    if mysql_host and mysql_user and mysql_password and mysql_database:
        db_uri = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}"
        st.sidebar.success("✅ Connected to MySQL Database")
    else:
        st.sidebar.error("❌ Please enter all MySQL credentials.")

else:
    db_uri = LOCAL_DB
    st.sidebar.success("✅ Connected to SQLite 3 Database")

# API Key Input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Validate API Key
if not api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API Key to proceed.")
    st.stop()

# LLM Model (Groq API)
llm = ChatGroq(
    temperature=0.5,
    groq_api_key=api_key,
    model_name="Llama3-8b-8192",
    streaming=True
)

# SQLite Connection Creator
def sqlite_creator():
    db_file_path = (Path(__file__).parent / "STUDENTS.db").absolute()
    return sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)

# Configure Database Connection
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_database=None):
    if db_uri == LOCAL_DB:
        return SQLDatabase(engine=create_engine("sqlite:///", creator=sqlite_creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_database and mysql_password and mysql_user):
            st.error("❌ Please enter all MySQL details.")
            st.stop()
        return SQLDatabase.from_uri(f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}")

# Establish Database Connection
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_database)
else:
    db = configure_db(db_uri)

# SQL Agent Toolkit & Agent Setup
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm, 
    toolkit=toolkit, 
    verbose=True, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Chat History
if "messages" not in st.session_state or st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot that can retrieve SQL queries for you. How can I help?"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Query Input
user_query = st.chat_input(placeholder="Enter your SQL query...")

# Process User Query
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        
        # Store response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
