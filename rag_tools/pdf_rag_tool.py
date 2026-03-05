import gradio as gr
import io
import os
import pandas as pd
import pymupdf as fitz

from langchain_classic.chains import RetrievalQA, LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from PyPDF2 import PdfReader


CHUNK_SIZE = 300
CHUNK_OVERLAP=50
MODEL_NAME = "deepseek-r1:latest"
EMBEDDING_MODEL = "embeddinggemma"
TEMPERATURE = 0.4

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

PROMPT = PromptTemplate(
    template="""Context: {context}

Question: {question}

Answer the question concisely based on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

If the question is about images or tables, refer to them specifically in your answer.""",
    input_variables=["context", "question"]
)


def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_text(text)
    
    return texts


def extract_images_and_tables(pdf_file):
    doc = fitz.open(pdf_file)
    images = []
    tables = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            
            images.append((f"Page {page_num + 1}, Image {idx + 1}", image))
            
        tables_on_page = page.find_tables()
        for idx, table in enumerate(tables_on_page):
            df = pd.DataFrame(
                table.extract()
            )
             
            tables.append((f"Page {page_num + 1}, Table {idx + 1}", df))
    
    return images, tables


def create_embeddings_and_vectorstore(texts):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    return vectorstore


def expand_query(query, llm):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""Given the following query, generate 3-5 related terms or phrases that could be relevant to the query. 
        Separate the terms with commas.
        
        Query: {query}
        
        Related terms:"""
    )
    chain = prompt | llm
    
    response = chain.invoke({"query": query})
    expanded_terms = [term.strip() for term in str(StrOutputParser().invoke(response)).split(",")]
    expanded_query = f"{query} {' '.join(expanded_terms)}"
    
    return expanded_query


def rag_pipeline(query, qa_chain, vectorstore):
    expanded_query = expand_query(query, llm)
    relavent_docs = vectorstore.similarity_search_with_score(expanded_query, k=5)
    
    log = ""
    log += "Query Expansion:\n"
    log += f"Original Query: {query}\n"
    log += f"Expanded Query: {expanded_query}\n"
    log += "\nRelevant Chunks:\n"
    for idx, (doc, score) in enumerate(relavent_docs, 1):
        log += f"Chunk {idx} (Score: {score:.4f}):\n Sample: {doc.page_content[:150]}...\n\n"
    
    response = qa_chain.invoke({"query": query})
    
    return response, log


def process_pdf_and_query(pdf_file, query):
    texts = process_pdf(pdf_file)
    images, tables = extract_images_and_tables(pdf_file)
    vectorstore = create_embeddings_and_vectorstore(texts)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    response, log = rag_pipeline(query, qa, vectorstore)
    
    return response["result"], len(texts), len(images), len(tables), log


def gradio_interface(pdf_file, query):
    result, num_chunks, num_images, num_tables, chunks_log = process_pdf_and_query(pdf_file.name, query)
    
    log = f"PDF processed successfully.\n"
    log += f"Number of text chunks: {num_chunks}\n"
    log += f"Number of images: {num_images}\n"
    log += f"Number of tables: {num_tables}\n\n"
    log += chunks_log
    
    return result, log


if __name__ == "__main__":
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.File(label="Upload PDF", file_types=[".pdf"]),
            gr.Textbox(label="Enter your question")
        ],
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Textbox(label="Processing Log")
        ],
        title="PDF Question Answering with RAG",
        description="Upload a PDF document and ask questions about its content."
    )
    
    iface.launch()
