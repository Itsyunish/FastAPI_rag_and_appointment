from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from practise.llm.splitter import get_splitter
from practise.utils.file_utils import ensure_upload_dir
from practise.db.mongo import metadata_collection
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from practise.llm.gemini_llm import get_llm
from practise.config import settings
from pathlib import Path
from langchain_core.runnables import Runnable, RunnableMap
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from practise.config import PINECONE_API_KEY, INDEX_HOST
import os
from pinecone import Pinecone

router = APIRouter()
UPLOAD_DIR = Path("uploads")
ensure_upload_dir(UPLOAD_DIR)
load_dotenv()

@router.post("/upload_file/")
async def upload_file(
    file: UploadFile = File(...),
    splitter: str = Query("recursive"),
    query: str = Query(...)
):
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

    # Save metadata
    metadata_collection.insert_one({
        "filename": file.filename,
        "file_type": ext[1:],
        "size_bytes": len(contents),
        "num_documents": len(docs),
        "num_chunks": len(texts),
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

        chunks_to_upsert = [
            {"id": f"{file.filename}_{i}", "text": doc.page_content}
            for i, doc in enumerate(texts)
        ]

        print("printing first chunk")
        print(chunks_to_upsert[0])

        await index.upsert_records(namespace="default", records=chunks_to_upsert)

        results = await index.search(
            namespace="default",
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

        # Combine context text for LLM
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
    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    
    
#     index_name = "multilingual-e5-large"
#     index = pc.Index(host=INDEX_HOST)
    
    
    
#     # 1. Create index if not exists
#     if not pc.has_index(index_name):
#         pc.create_index_for_model(
#             name=index_name,
#             cloud="aws",
#             region="us-east-1",
#             embed={"model": "multilingual-e5-large", "field_map": {"text": "text"}}
#         )
        
#     chunks_to_upsert = []

#     for i, doc in enumerate(texts):
#         chunks_to_upsert.append({
#             "id": f"{file.filename}_{i}",
#             "text": doc.page_content,  
#         })

#     index.upsert_records(namespace="default", records=chunks_to_upsert)



#     print("printing chunks")
#     print(chunks_to_upsert[0])
    
        
    
#     results = index.search(
#         namespace="default",
#         query={
#             "inputs": {"text": query},
#             "top_k": 3, 
#             # "filter": {"filename": file.filename}
#             }
#     )
    
#     print("printing result")
#     print(f"{results}\n\n")
    
#     print("looping through result")
#     for hit in results['result']['hits']:
#         print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | text: {hit.get('fields', {}).get('text', '')[:50]}")


#     context_text = "\n".join(
#     hit.get("fields", {}).get("text", "")  # get 'text' from fields
#     for hit in results['result']['hits']
# )



#     print("context text")
#     print(context_text)

#     llm = get_llm()

#     chain = prompt | llm | StrOutputParser()

#     answer = chain.invoke({"context": context_text, "query": query})

#     return {"query": query, "answer": answer}
