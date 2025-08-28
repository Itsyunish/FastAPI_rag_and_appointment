from fastapi import APIRouter, HTTPException
from practise.db.redis_client import r, get_conversation
import json

router = APIRouter()

@router.get("/conversation_history/")
def get_all_conversations():
    try:
        keys = r.keys("*")
        conversations = {key.decode(): json.loads(r.get(key)) for key in keys}
        return {"total_sessions": len(keys), "conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")
