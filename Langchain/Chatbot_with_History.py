import os
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
api_key=os.environ["GROQAI_API_KEY"]=os.getenv("GROQAI_API_KEY")

embeddings=OllamaEmbeddings(model="gemma:2b")

# Streamlit Application
st.title("RAG using Ollama Embeddings and Groq LLM with Chat History")
st.write("This is a simple RAG application using Ollama Embeddings and Groq LLM with Chat History")
api_key=st.text_input("Enter your Groq API Key: ", type="password")
if api_key:
    llm=ChatGroq(temperature=0.5,model="Llama3-8b-8192",api_key=api_key)
    session_id=st.text_input("Session ID",value="session_one")
    if 'store' not in st.session_state:
        st.session_state.store={}
    upload_file=st.file_uploader("Upload a PDF file", type="pdf",accept_multiple_files=True)

    # Creating a temporary file path 
    if upload_file:
        documents=[]
        for file in upload_file:
            temp_pdf="./temp.pdf"
            with open(temp_pdf,"wb") as f:
                f.write(file.getvalue())
                file_name=file.name
            loader=PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_documents=text_splitter.split_documents(documents)
        db=Chroma.from_documents(final_documents, embeddings,persist_directory="./chroma_db")
        retriever=db.as_retriever()

        system_prompt=""" 
        Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without reference to the chat history.
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        """
        prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        history_retriever=create_history_aware_retriever(llm, retriever, prompt)

        # Creating another prompt for Q and A
        system_prompt_2=""" 
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        \n\n
        {context}
        """

        qa_prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt_2),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        qa_chain=create_stuff_documents_chain(llm, qa_prompt)
        rag_chain=create_retrieval_chain(history_retriever, qa_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Enter your query from uploaded PDF")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke({"input":user_input}, {"config": {"configurable": {"session_id": session_id}}})
            st.write(st.session_state.store)
            st.success("Assistant: ",response["answer"])
            st.write("Chat History: ",session_history.messages)
            # With Streamlit Expander
            with st.expander("Document Similarity Search"):
                for i,doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
    else:
        st.warning("Please upload a PDF file or Please provide GROQ API Key")