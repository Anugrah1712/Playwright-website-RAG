from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from preprocess import preprocess_vectordbs
from inference import inference
import validators
import uvicorn
import json
from webscrape import scrape_web_data

app = FastAPI()

# Enable CORS for frontend (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this if your frontend URL changes
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Store session state
session_state = {
    "retriever": None,
    "preprocessing_done": False,
    "index": None,
    "docstore": None,
    "embedding_model_global": None,
    "pinecone_index": None,
    "vs": None,
    "selected_vectordb": None,
    "selected_chat_model": None,
    "messages": []
}

@app.post("/preprocess")
async def preprocess(
    doc_files: List[UploadFile] = File(...),
    links: str = Form(...),
    embedding_model: str = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...)
):
    """ Preprocessing: Handle document uploads and web scraping """

    try:
        print("\n🔍 Preprocessing Started...")
        print(f"📂 Received {len(doc_files)} document(s)")
        print(f"🔗 Links received: {links}")
        print(f"📊 Embedding Model: {embedding_model}")
        print(f"🔢 Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")

        # Validate links
        links_list = json.loads(links)
        for link in links_list:
            if not validators.url(link):
                raise HTTPException(status_code=400, detail=f"❌ Invalid URL: {link}")

        # Validate uploaded files
        if not doc_files:
            raise HTTPException(status_code=400, detail="❌ No documents uploaded!")

        for file in doc_files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="❌ One of the uploaded files is empty!")

        # Web scraping
        try:
            scraped_content = await scrape_web_data(links_list)
            print("✅ Web Scraping Completed!\n")
            print(json.dumps(scraped_content, indent=2))  # Log scraped data

        except Exception as e:
            print(f"❌ Web scraping error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Web Scraping failed: {str(e)}")

        # Process documents
        try:
            result = await preprocess_vectordbs(
                doc_files, links_list, embedding_model, chunk_size, chunk_overlap
            )
            session_state["preprocessing_done"] = True  # Mark preprocessing as done
            return {
                "status": "success",
                "message": "Preprocessing completed successfully!",
                "scraped_content": scraped_content
            }

        except Exception as e:
            print(f"❌ Error in preprocess_vectordbs: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")


@app.post("/select_vectordb")
async def select_vectordb(vectordb: str = Form(...)):
    """ Set selected vector database """
    session_state["selected_vectordb"] = vectordb
    print(f"✅ Selected Vector Database: {vectordb}\n")
    return {"message": f"Selected Vector Database: {vectordb}"}


@app.post("/select_chat_model")
async def select_chat_model(chat_model: str = Form(...)):
    """ Set selected chat model """
    session_state["selected_chat_model"] = chat_model
    print(f"✅ Selected Chat Model: {chat_model}\n")
    return {"message": f"Selected Chat Model: {chat_model}"}


@app.post("/chat")
async def chat_with_bot(prompt: str = Form(...)):
    """ Chatbot interaction """
    if not session_state["preprocessing_done"]:
        raise HTTPException(status_code=400, detail="❌ Preprocessing must be completed before inferencing.")

    if not session_state["selected_vectordb"] or not session_state["selected_chat_model"]:
        raise HTTPException(status_code=400, detail="❌ Please select both a vector database and a chat model before chatting.")

    # Store user message
    session_state["messages"].append({"role": "user", "content": prompt})

    # Run inference
    try:
        response = inference(
            session_state["selected_vectordb"],
            session_state["selected_chat_model"],
            prompt,
            session_state["retriever"],
            session_state["embedding_model_global"],
            session_state["index"],
            session_state["docstore"],
            session_state["pinecone_index"],
            session_state["vs"],
            session_state["messages"]
        )

        # Store assistant response
        session_state["messages"].append({"role": "assistant", "content": response})

        print(f"🤖 Chatbot Response: {response}\n")
        return {"response": response}

    except Exception as e:
        print(f"❌ Error in inference: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


@app.post("/reset")
async def reset_chat():
    """ Reset chatbot history """
    session_state["messages"] = []
    print("🔄 Chat history reset.\n")
    return {"message": "Chat history reset."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
