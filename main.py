from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient
from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import Annotated
import mailtrap as mt
from dotenv import load_dotenv
import redis
import json
import os
from pathlib import Path

            
            
# Load environment variables
load_dotenv()

app = FastAPI(title="RAG API")

mailtrap_api_key = os.getenv("MAILTRAP_API_KEY")        


r = redis.Redis(host='localhost', port=6379, db=0)

def get_conversation(session_id: str):
    conv = r.get(session_id)
    if conv:
        return json.loads(conv)
    return []
    
def save_conversation(session_id: str, conversation: list):
    r.set(session_id, json.dumps(conversation))

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load LLM
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)


# splitting text
def get_splitter(splitter_type: str):
    if splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    elif splitter_type == "markdown":
        return MarkdownTextSplitter()
    else:
        raise ValueError("Invalid splitter type. Choose 'recursive' or 'markdown'.")


# Embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
def get_vector_store(texts, index_name="pdf-reader"):
    import pinecone
    from pinecone import ServerlessSpec

    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    vector_store = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=get_embeddings(),
        index_name=index_name
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# RAG Chain
def get_rag_chain(llm, retriever):
    system_prompt = """
    You are a helpful assistant. Use the following context to answer the question:
    {context}

    - Answer only using the context.
    - If the context does not have the answer, say "I don't know."
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


client = MongoClient("mongodb://localhost:27017/")
db = client["rag_db"]
collection = db["metadata"]



# API Endpoint
@app.post("/upload_file/")
async def upload_file(
    file: UploadFile = File(..., description="Upload a PDF or TXT file"),
    splitter: str = Query("recursive", description="Select splitter: 'recursive' or 'markdown'")
):
    # Validate file type
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and text files are supported.")

    file_path = UPLOAD_DIR / file.filename

    try:
        # Read file asynchronously and save
        contents = await file.read()
        file_path.write_bytes(contents)
        size = len(contents)

        # Load documents
        ext = file_path.suffix.lower()
        if ext == ".txt":
            loader = TextLoader(str(file_path))
            docs = loader.load()
        elif ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # Split documents into chunks
        text_splitter = get_splitter(splitter)
        texts = text_splitter.split_documents(docs)
        
        metadata = {
            "filename": file.filename,
            "file_type": ext[1:],
            "size_bytes": size,
            "num_documnets": len(docs),
            "num_chunks": len(texts),
            "splitter": splitter
            
        }
        
        collection.insert_one(metadata)

        # Create retriever and RAG chain
        retriever = get_vector_store(texts)
        rag_chain = get_rag_chain(get_llm(), retriever)
        app.state.rag_chain = rag_chain

        return {
            "filename": file.filename,
            "documents": len(docs),
            "chunks": len(texts),
            "message": f"{ext[1:].upper()} uploaded and processed successfully."
        }

    except Exception:
        import traceback
        raise HTTPException(status_code=500, detail=f"Processing failed:\n{traceback.format_exc()}")
    
    
@app.get("/conversation_query/")
async def conversation_query(
    session_id: str = Query(..., description="Session ID for the conversation"),
    q: str = Query(..., description="Your question")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_chain = getattr(app.state, "rag_chain", None)
    if not rag_chain:
        raise HTTPException(status_code=400, detail="No file has been uploaded yet. Use /upload_file first.")

    try:
        # Load conversation history from Redis
        conversation_history = get_conversation(session_id)

        # Append new user query
        conversation_history.append({"role": "user", "content": q})

        # This is what we send to RAG so it sees the full chat history
        conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])

        # Get answer from RAG
        result = rag_chain.invoke({"input": conversation_text})
        answer = result.get("answer") or result.get("output_text") or result.get("text") or result.get("output")

        # Save assistant reply to history
        conversation_history.append({"role": "assistant", "content": answer})
        save_conversation(session_id, conversation_history)

        return {
            "session_id": session_id,
            "query": q,
            "answer": answer,
            "history": conversation_history  
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error generating answer:\n{traceback.format_exc()}")



@app.get("/query/")
async def query_rag(q: str):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_chain = getattr(app.state, "rag_chain", None)
    if not rag_chain:
        raise HTTPException(status_code=400, detail="No file has been uploaded yet. Use /upload_file first.")

    try:
        result = rag_chain.invoke({"input": q})
        return {"query": q, "answer": result.get("answer") or result.get("output_text")}
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error generating answer:\n{traceback.format_exc()}")
    
    
def format_size_mb(size_bytes: int) -> str:
    """Convert bytes to MB."""
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"


@app.get("/get_metadata/")
async def get_metadata():
    try:
        records = list(collection.find({}, {"_id": 0}))
        
        for data in records:
            data["size"] = format_size_mb(data["size_bytes"])
        
        return JSONResponse(content={"metadata": records})
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error fetching metadata:\n{traceback.format_exc()}")
    
    
@app.get("/conversation_history/")
async def get_all_conversations():
    try:
        keys = r.keys("*")
        conversations = {}
        for key in keys:
            conversations[key] = json.loads(r.get(key))
        return {"total_sessions": len(keys), "conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching all conversations: {str(e)}")
    


appointment_db = client["appointments_db"]
appointment_collection = appointment_db["appointment"]


class UserInfo(BaseModel):
    name: Annotated[str, Field(max_length=50, description="Name of the user")]
    email: Annotated[EmailStr, Field(description="Email of the user", example="xyz@gmail.com")]
    appointment_date: str  
    appointment_time: str  

    @field_validator("name")
    def validate_name(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        return v

    @field_validator("appointment_date")
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date format. Expected YYYY-MM-DD")
        return v

    @field_validator("appointment_time")
    def validate_time(cls, v):
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError("Invalid time format. Expected HH:MM (24-hour)")
        return v

# Endpoint
@app.post("/schedule_appointment/")
def schedule_appointment(info: UserInfo):
      
        try:

            metadata = {
                "name": info.name,
                "email": info.email,
                "appointment_date": info.appointment_date,
                "appointment_time": info.appointment_time,
              }
            
                                     
            mail = mt.Mail(
                sender=mt.Address(email="hello@demomailtrap.co", name="Mailtrap Test"),
                to=[mt.Address(email=info.email, name=info.name)],
                subject="Your Appointment Details!",
                text=(
                    f"Hello {info.name}, \n\n"
                    f"Your appointment is scheduled on {info.appointment_date} at {info.appointment_time}.\n\n"
                    f"Thank you!"
                ),
                category="Appointment Notification",
            )

            client = mt.MailtrapClient(token=mailtrap_api_key)
            response = client.send(mail)
            
            
            metadata["mailtrap"] = {
                "success": response["success"],
                "mailtrap_message_id": response["message_ids"][0]
            }
            
            appointment_collection.insert_one(metadata)

            
            return JSONResponse(content={
                "message": "Appointment scheduled successfully",
                "appointment_details": {
                    "name": info.name,
                    "email": info.email,
                    "appointment_date": info.appointment_date,
                    "appointment_time": info.appointment_time
            },
                "mailtrap_status": response
        })
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": f"Information saving failed: {str(e)}"}
            )
            
   

            