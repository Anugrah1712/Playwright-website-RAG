from playwright.sync_api import sync_playwright
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import numpy as np
import os

# Ensure that the embedding model is initialized
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load API Key securely
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if TOGETHER_API_KEY is None:
    raise ValueError("Together API Key is missing. Set it as an environment variable.")

# Ensure embedding model is initialized
embedding_model_global = OpenAIEmbeddings()

def inference_chroma(chat_model, question, retriever, chat_history):
    chat_model = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert financial advisor. Use the context and chat history to answer accurately.\n\n"
            "Context: {context}\n\n"
            "{question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        ),
    )

    retriever_chain = retriever | prompt_template | chat_model | StrOutputParser()
    result = retriever_chain.invoke(question_with_history)
    print(result)
    return result


def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history):
    if embedding_model_global is None:
        raise ValueError("Embedding model is not initialized.")

    chat_model = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    query_embedding = embedding_model_global.embed_query(question)
    D, I = index.search(np.array([query_embedding]), k=1)

    doc_id = I[0][0]
    document = docstore.search(doc_id)
    context = document.page_content

    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=(
            "You are an expert financial advisor. Use the context and chat history to answer questions accurately.\n"
            "Chat History:\n{history}\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    qa_chain = prompt_template | chat_model | StrOutputParser()
    answer = qa_chain.invoke({"history": history_context, "context": context, "question": question})
    print(answer)

    return answer


def inference_qdrant(chat_model, question, embedding_model_global, client, chat_history):
    if embedding_model_global is None:
        raise ValueError("Embedding model is not initialized.")

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    query_embedding = embedding_model_global.embed_query(question_with_history)

    search_results = client.search(
        collection_name="text_vectors",
        query_vector=query_embedding.tolist(),
        limit=2
    )

    contexts = [result.payload['page_content'] for result in search_results]
    context = "\n".join(contexts)

    prompt = f"""
    You are a helpful assistant. Use the retrieved documents to answer:
    
    Context:
    {context}

    {question_with_history}

    Answer:
    """

    chat_model = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        model=chat_model
    )

    response = chat_model.predict(prompt)
    print(response)
    return response


def inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history):
    if embedding_model_global is None:
        raise ValueError("Embedding model is not initialized.")

    query_embedding = embedding_model_global.embed_query(question)

    search_results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=2,
        include_metadata=True
    )

    contexts = [result['metadata']['text'] for result in search_results['matches']]
    context = "\n".join(contexts)

    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    prompt = f"""
    You are a helpful assistant. Use the retrieved documents and chat history to answer:
    
    Chat History:
    {formatted_history}

    Context:
    {context}

    Question: {question}
    
    Answer:
    """

    chat_model = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        model=chat_model
    )

    response = chat_model.predict(prompt)
    print(response)
    return response


def inference_weaviate(chat_model, question, vs, chat_history):
    chat_model = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    template = """
    You are an expert financial advisor. Use the context and chat history to answer accurately:

    Context:
    {context}

    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    retriever = vs.as_retriever()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()

    result = rag_chain.invoke(question_with_history)
    print(result)
    return result


def inference(vectordb_name, chat_model, question, retriever, embedding_model_global, index, docstore, pinecone_index, vs, chat_history):
    if embedding_model_global is None:
        embedding_model_global = OpenAIEmbeddings()

    if vectordb_name == "Chroma":
        return inference_chroma(chat_model, question, retriever, chat_history)
    elif vectordb_name == "FAISS":
        return inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history)
    elif vectordb_name == "Qdrant":
        return inference_qdrant(chat_model, question, embedding_model_global, vs, chat_history)
    elif vectordb_name == "Pinecone":
        return inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history)
    elif vectordb_name == "Weaviate":
        return inference_weaviate(chat_model, question, vs, chat_history)
    else:
        print("Invalid Choice")
