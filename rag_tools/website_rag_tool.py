import numpy as np
import os
import requests
import streamlit as st
import tempfile
import traceback

from langchain_classic.chains import RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MODEL_NAME = "deepseek-r1:latest"
EMBEDDING_MODEL = "embeddinggemma"
TEMPERATURE = 0.4


if "qa" not in st.session_state:
    st.session_state.qa = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def fetch_and_process_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        with st.spinner("Fetching and processing the webpage..."):
            response = requests.get(url, headers=headers )
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html", encoding="utf-8") as tmp_file:
                tmp_file.write(response.content.decode('utf-8', errors='ignore'))
                tmp_file_path = tmp_file.name
                
            try:
                loader = BSHTMLLoader(tmp_file_path, open_encoding='utf-8', bs_kwargs={'features': 'lxml', 'from_encoding': 'utf-8'})
                documents = loader.load()
            except ImportError:
                st.warning("Falling back to built-in html parser.")
                loader = BSHTMLLoader(tmp_file_path, open_encoding='utf-8', bs_kwargs={'features': 'html.parser', 'from_encoding': 'utf-8'})
                documents = loader.load()
            
            os.unlink(tmp_file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            return texts
    
    except Exception as e:
        tb = traceback.TracebackException.from_exception(e)
        st.error(f"Error fetching or processing the webpage: {e}")
        st.error("\n".join(tb.format()))
        return None
    

def initialize_rag_pipeline(texts):
    with st.spinner("Initializing RAG pipeline..."):
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE
        )
        
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        template = """Context: {context}

        Question: {question}

        Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

        But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa, vectorstore
    

if __name__ == "__main__":
    st.title("RAG Website Query Tool")
    st.write("Enter a URL to fetch and process the webpage, then ask questions about its content.")
    
    url = st.text_input("Enter URL")
    
    if st.button("Fetch and Process") and url:
        texts = fetch_and_process_url(url)
        
        if texts:
            st.success(f"Successfully fetched and processed the webpage. Total chunks created: {len(texts)}")
            st.session_state.qa, st.session_state.vectorstore = initialize_rag_pipeline(texts)
            st.session_state.chat_history = []
    
    if st.session_state.qa and st.session_state.vectorstore:
        st.write("---")
        st.subheader("Ask a question about the webpage")
        
        query = st.text_input("Enter your question here")
        
        if st.button("Ask") and query:
            with st.spinner("Generating answer..."):
                relavent_docs = st.session_state.vectorstore.similarity_search_with_score(query, k=5)
                
                with st.expander("Relevant Chunks"):
                    for idx, (doc, score) in enumerate(relavent_docs, 1):
                        st.write(f"Chunk {idx} (Score: {score:.4f})")
                        st.write(doc.page_content[:200] + "...")
                        st.write("---")
                
                response = st.session_state.qa.invoke({"query": query})
                
                st.session_state.chat_history.append({"q": query, "a": response["result"]})
                
            st.write("---")
            st.subheader("Chat History")
            for chat in st.session_state.chat_history:
                st.write(f"**Question:** {chat['q']}")
                st.write(f"**Answer:** {chat['a']}")
                st.write("---")
        
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This is a RAG (Retrieval-Augmented Generation) system that allows you to:
        1. Input any website URL
        2. Process its content
        3. Ask questions about the content
        
        The system uses:
        - Ollama (deepseek-r1) for text generation
        - FAISS for vector storage and retrieval
        - LangChain for RAG pipeline
        """)
        
        st.subheader("Model Configuration")
        st.write(f"Model: {MODEL_NAME}")
        st.write(f"Temperature: {TEMPERATURE}")
        st.write(f"Chunk Size: {CHUNK_SIZE}")
        st.write(f"Chunk Overlap: {CHUNK_OVERLAP}")
        st.write(f"Embedding Model: {EMBEDDING_MODEL}")
