from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from practise.llm.splitter import get_splitter
from practise.utils.file_utils import ensure_upload_dir
from practise.db.mongo import metadata_collection
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from practise.llm.gemini_llm import get_llm
from practise.config import settings
from pathlib import Path
from practise.utils.file_utils import format_size_mb
import uuid
from langchain_core.runnables import Runnable, RunnableMap
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
import os

router = APIRouter()
UPLOAD_DIR = Path("uploads")
ensure_upload_dir(UPLOAD_DIR)
load_dotenv()

@router.post("/upload_file/")
async def upload_file(
    file: UploadFile = File(...),
    splitter: str = Query(default="recursive"),
    user_id: int = Query(...,gt=299,lt=4001,description="range(300-4000)")
    
):

    existing_user = metadata_collection.find_one({"user_id": user_id})
    if existing_user:
        raise HTTPException(status_code=400, detail="User id already exists")
        
    
    
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and text files are supported.")
    

    file_path = UPLOAD_DIR / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    # Load document
    ext = file_path.suffix.lower()
    loader = TextLoader(str(file_path)) if ext == ".txt" else PyMuPDFLoader(str(file_path))
    docs = loader.load()
    

    # Split document into chunks
    splitter_instance = get_splitter(splitter)
    texts = splitter_instance.split_documents(docs)
    

    if not texts:
        raise HTTPException(status_code=400, detail="Upload the file for processing")
    
    namespace_id = str(uuid.uuid4()) 

    # Save metadata
    metadata_collection.insert_one({
        "namespace_id": namespace_id,
        "user_id": user_id,
        "filename": file.filename,
        # "file_type": ext[1:],
        "file_size": format_size_mb(len(contents)),
        # "num_documents": len(docs),
        # "num_chunks": len(texts),
        "splitter": splitter
    })

    
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    
    index_name = "multilingual-e5-large"

    async with pc.IndexAsyncio(host=settings.index_host.get_secret_value()) as index:

        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={"model": "multilingual-e5-large", "field_map": {"text": "text"}}
            )
            
            
        try:
            chunks_to_upsert = [
                {"id": f"{file.filename}_{i}", "text": doc.page_content}
                for i, doc in enumerate(texts)
            ]

            print("printing first chunk")
            print(chunks_to_upsert[0])
            
            await index.upsert_records(namespace=namespace_id, records=chunks_to_upsert)

            return {"message": "File uploaded successfully in vectorstore"}

        except Exception:
            
            raise HTTPException(status_code=500, details="File has not been stored in vectorstore")
            


   
   
@router.post("/query/")
async def query(query:str =  Query(...,description="ask the question for user query"),
                user_id:int = Query(...,description="enter the same user id as while uploading file")):
    
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    
    user_match = metadata_collection.find_one({"user_id": user_id})
    
    if user_match:
        namespace_id = user_match.get("namespace_id")
        print(f"Namespace id found:{namespace_id}")
    else:
        print("User not found")
        
    async with pc.IndexAsyncio(host=settings.index_host.get_secret_value()) as index:

        results = await index.search(
                namespace=namespace_id,
                query={
                    "inputs": {"text": query},
                    "top_k": 3,
                }
            )

        print("looping through results")
        for hit in results['result']['hits']:
            print(
                f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | text: {hit.get('fields', {}).get('text', '')[:50]}"
            )

        context_text = "\n".join(
            hit.get("fields", {}).get("text", "") for hit in results['result']['hits']
        )

        print("context text")
        print(context_text)

        # 6. Query LLM
        llm = get_llm()
        prompt_template = """
        You are a helpful assistant. Use the following context to answer the user's query.
        Don't give wrong answers. If you don't know say it in a polite way.

        Context:
        {context}

        Query:
        {query}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        chain: Runnable = RunnableMap({
        "query": lambda x: x['query'],
        "context": lambda x: context_text,  
         }) | prompt | llm | StrOutputParser()

        answer = await chain.ainvoke({'query': query})


        return {"query": query, "answer": answer}
    
   
   
    
   
   
   
   
   
   
   
   
   
   
   
   
    