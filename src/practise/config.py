import os
from dotenv import load_dotenv

load_dotenv()

MAILTRAP_API_KEY = os.getenv("MAILTRAP_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
