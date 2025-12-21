from flask import Flask
from flask_pymongo import PyMongo
import os

app = Flask(__name__)
# Use the same URI as app.py
app.config["MONGO_URI"] = "mongodb+srv://jagadeesh:8074563501@cluster0.ur44l.mongodb.net/examproctordb?retryWrites=true&w=majority"
mongo = PyMongo(app)

def test_connection():
    print("Testing connection to MongoDB Atlas...")
    try:
        # Attempt to list collections to verify connection
        print("Collections:", mongo.db.list_collection_names())
        
        # Check for students
        print("\nChecking 'students' collection:")
        students = list(mongo.db.students.find())
        if not students:
            print("No students found in database!")
        else:
            print(f"Found {len(students)} students:")
            for s in students:
                print(f"- Name: {s.get('Name')}, Email: {s.get('Email')}, Role: {s.get('Role')}, Password: {s.get('Password')}")
                
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    with app.app_context():
        test_connection()
