import os
import sys
from fastapi import FastAPI
from langserve import add_routes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
api_key=os.environ["GROQAI_API_KEY"]=os.getenv("GROQAI_API_KEY")

llm=ChatGroq(groq_api_key=api_key,model_name="mixtral-8x7b-32768")

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the given sentences into {language}"),
    ("human", "{text}")  # Placeholder for user input
])

parser=StrOutputParser()

chain=prompt|llm|parser

# App Definition
app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain"
)

# Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)