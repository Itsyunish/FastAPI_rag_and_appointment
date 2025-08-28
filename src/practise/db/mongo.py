from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

# Collections
metadata_collection = client["rag_db"]["metadata"]
appointment_collection = client["appointments_db"]["appointment"]
