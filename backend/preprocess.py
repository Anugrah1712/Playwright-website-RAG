#preprocess.py 
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from fastapi import UploadFile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore  
from langchain_community.vectorstores import FAISS
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


#from playwright.async_api import async_playwright

async def preprocess_text(files: list[UploadFile], links, size, overlap):
    import time
    
    paragraphs = []

    # Step 1: Process each file
    for file in files:
        if file.filename.endswith(".pdf"):
            await file.seek(0)
            reader = PdfReader(file.file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    paragraphs.extend(page_text.split("\n"))
        elif file.filename.endswith(".docx"):
            await file.seek(0)
            docx = DocxDocument(file.file)
            for paragraph in docx.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)

        # # Step 2: Use Playwright for web scraping
        # async with async_playwright() as p:
        #     browser = await p.chromium.launch(headless=True)  # <-- Await launch()
        #     context = await browser.new_context()  # <-- Await new_context()
        #     page = await context.new_page()  # <-- Await new_page()

        #     for link in links:
        #         try:
        #             await page.goto(link, timeout=60000)  # <-- Await page navigation
        #             await asyncio.sleep(3)  # Allow page to load
        #             body_text = await page.text_content("body")  # <-- Await text extraction
        #             paragraphs.extend(body_text.split("\n"))

        #         except Exception as link_error:
        #             print(f"Failed to process link {link}: {link_error}")

        #     await browser.close()  # <-- Await browser close


    # Step 3: Remove empty paragraphs
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    # Step 4: Convert paragraphs to Document objects
    docs = [LangchainDocument(page_content=para) for para in paragraphs]

    # Step 5: Use RecursiveCharacterTextSplitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    text_chunks = text_splitter.split_documents(docs)

    return text_chunks

def preprocess_chroma(text, embedding_model_name, persist_directory):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma.from_documents(documents=text, embedding=embedding_model, persist_directory=persist_directory)
    vectordb.persist()
    
    retriever = vectordb.as_retriever()
    return vectordb, retriever

def preprocess_faiss(text, embedding_model_name):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    import numpy as np
    import faiss
    from langchain.docstore.in_memory import InMemoryDocstore
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    texts = [doc.page_content for doc in text]
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    docstore = InMemoryDocstore({i: Document(page_content=texts[i]) for i in range(len(texts))})
    index_to_docstore_id = {i: i for i in range(len(texts))}

    vector_store = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model.embed_query
    )

    return index, docstore, index_to_docstore_id, vector_store

def preprocess_weaviate(text, embedding_model_name):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    import os
    import weaviate
    from weaviate.auth import AuthApiKey
    from langchain_weaviate.vectorstores import WeaviateVectorStore

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    weaviate_url = os.getenv("WEAVIATE_URL", "https://pdarzyhgqows9ocn5oava.c0.asia-southeast1.gcp.weaviate.cloud")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "u5qU5QbMDcvKw8pPewXXAcmHNvRYNstmOxES")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(weaviate_api_key),

    )

    vs = WeaviateVectorStore.from_documents(
        documents=text,
        embedding=embedding_model,
        client=client
    )
    client.close()

    return vs

def preprocess_pinecone(text, embedding_model_name):
    import numpy as np
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from pinecone import Pinecone, ServerlessSpec
    import uuid

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    texts = [doc.page_content for doc in text]
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings).tolist()

    pinecone = Pinecone(
        api_key="pcsk_42Yw14_EaKdaMLiAJfWub3s2sEJYPW3jyXXjdCYkH8Mh8rD8wWJ3pS6oCCC9PGqBNuDTuf",
        environment="us-east-1"
    )

    index_name = "test5"
    indexes = pinecone.list_indexes().names()

    if index_name in indexes:
        pinecone.delete_index(index_name)

    pinecone.create_index(
        name=index_name,
        dimension=len(embeddings[0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled"
    )

    pinecone_index = pinecone.Index(index_name)

    upsert_data = [(str(uuid.uuid4()), embeddings[i], {"text": texts[i]}) for i in range(len(texts))]

    batch_size = 100  
    for i in range(0, len(upsert_data), batch_size):
        batch = upsert_data[i: i + batch_size]
        pinecone_index.upsert(vectors=batch)

    return pinecone_index

from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import UploadFile
import sys
import sqlite3

sys.modules["sqlite3"] = sqlite3

embedding_model_global = None

async def preprocess_vectordbs(files: list[UploadFile], links, embedding_model_name, size, overlap):
    global embedding_model_global

    from preprocess import preprocess_text, preprocess_chroma, preprocess_faiss, preprocess_weaviate, preprocess_pinecone #import local functions

    text = await preprocess_text(files, links, size, overlap)
    # print(text)
    persist_directory = 'db'

    embedding_model_global = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    vectordb, retriever = preprocess_chroma(text, embedding_model_name, persist_directory)
    print("chroma db processing done")
    index, docstore, index_to_docstore_id, vector_store = preprocess_faiss(text, embedding_model_name)
    print("faiss db processing done")
    vs = preprocess_weaviate(text, embedding_model_name)
    print("weaviate db processing done")
    pinecone_index = preprocess_pinecone(text, embedding_model_name)
    print("pinecone db processing done")

    return index, docstore, index_to_docstore_id, vector_store, retriever, pinecone_index, embedding_model_global, vs