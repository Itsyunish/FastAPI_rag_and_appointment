from pymongo import MongoClient

client = MongoClient("")

# Collections
metadata_collection = client["test_db"]["user"]
appointment_collection = client["test_db"]["appointment"]
