from fastapi import APIRouter, Query, HTTPException, Request
from practise.db.redis_client import get_conversation, save_conversation

router = APIRouter()

@router.get("/conversation_query/")
async def conversation_query(request: Request, session_id: str = Query(...), q: str = Query(...)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_chain = getattr(request.app.state, "rag_chain", None)
    if not rag_chain:
        raise HTTPException(status_code=400, detail="No file uploaded. Upload first.")

    conversation_history = get_conversation(session_id)
    conversation_history.append({"role": "user", "content": q})

    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
    result = rag_chain.invoke({"input": conversation_text})
    answer = result.get("answer") or result.get("output_text") or result.get("text") or result.get("output")

    conversation_history.append({"role": "assistant", "content": answer})
    save_conversation(session_id, conversation_history)

    return {"session_id": session_id, "query": q, "answer": answer, "history": conversation_history}


@router.get("/query/")
async def query_rag(request: Request, q: str):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_chain = getattr(request.app.state, "rag_chain", None)
    if not rag_chain:
        raise HTTPException(status_code=400, detail="No file uploaded. Upload first.")

    result = rag_chain.invoke({"input": q})
    return {"query": q, "answer": result.get("answer") or result.get("output_text")}
