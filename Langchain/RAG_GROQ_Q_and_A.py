import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
api_key=os.environ["GROQAI_API_KEY"]=os.getenv("GROQAI_API_KEY")

# LLM
llm = ChatGroq(temperature=0, model_name="Llama3-8b-8192",api_key=api_key)
prompt=ChatPromptTemplate.from_template(""" 
Answer the following question based on the provided context.
please provide most accurate answer based on the question.
<context>
{context}
</context>
Question: {input}
""")

st.title("RAG using Ollama Embeddings and Groq LLM")

def vector_embedding():
   if "vectors" not in st.session_state:
      st.session_state.embeddings=OllamaEmbeddings(model="gemma:2b")
      st.session_state.loader=PyPDFLoader("C:\\Users\\DELL\\Generative_AI\\Langchain\\NIPS-2017-attention-is-all-you-need-Paper.pdf")
      st.session_state.docs=st.session_state.loader.load()
      st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
      st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
      st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt=st.text_input("Enter your query from uploaded PDF")
if st.button("Document Embedding"):
   vector_embedding()
   st.write("Embedding Completed")

import time
if user_prompt:
   document_chain=create_stuff_documents_chain(llm, prompt)
   retriever=st.session_state.vectors.as_retriever()
   retrieval_chain=create_retrieval_chain(retriever, document_chain)
   start=time.process_time()
   response=retrieval_chain.invoke({"input":user_prompt})
   print(f"Response time: {time.process_time()-start}")
   st.write(response["answer"])
   # With Streamlit Expander
   with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

   