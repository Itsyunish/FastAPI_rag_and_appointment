import redis
import json

r = redis.Redis(host="localhost", port=6379, db=0)

def get_conversation(session_id: str):
    conv = r.get(session_id)
    return json.loads(conv) if conv else []

def save_conversation(session_id: str, conversation: list):
    r.set(session_id, json.dumps(conversation))
