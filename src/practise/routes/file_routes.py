from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from practise.llm.splitter import get_splitter
from practise.llm.embeddings import get_embeddings
from practise.llm.rag_chain import get_llm, get_rag_chain
from practise.utils.file_utils import ensure_upload_dir
from practise.db.mongo import metadata_collection
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from pathlib import Path
import os
import pinecone
from pinecone import ServerlessSpec

router = APIRouter()
UPLOAD_DIR = Path("uploads")
ensure_upload_dir(UPLOAD_DIR)

@router.post("/upload_file/")
async def upload_file(request: Request, file: UploadFile = File(...), splitter: str = Query("recursive", description="Select splitter: 'recursive' or 'character'")):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and text files are supported.")

    file_path = UPLOAD_DIR / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    ext = file_path.suffix.lower()
    loader = TextLoader(str(file_path)) if ext == ".txt" else PyPDFLoader(str(file_path))
    docs = loader.load()

    splitter_instance = get_splitter(splitter)
    texts = splitter_instance.split_documents(docs)

    metadata = {
        "filename": file.filename,
        "file_type": ext[1:],
        "size_bytes": len(contents),
        "num_documents": len(docs),
        "num_chunks": len(texts),
        "splitter": splitter
    }
    metadata_collection.insert_one(metadata)

    # Vector store & RAG chain
    index_name = "pdf-reader"
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=384, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    vector_store = PineconeVectorStore.from_documents(documents=texts,
                                                      embedding=get_embeddings(),
                                                      index_name=index_name)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Save RAG chain in app.state
    rag_chain = get_rag_chain(get_llm(), retriever)
    request.app.state.rag_chain = rag_chain

    return {
        "filename": file.filename,
        "documents": len(docs),
        "chunks": len(texts),
        "message": f"{ext[1:].upper()} uploaded and processed successfully."
    }
