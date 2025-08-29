from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

# Collections
metadata_collection = client["test_db"]["user"]
appointment_collection = client["test_db"]["appointment"]
