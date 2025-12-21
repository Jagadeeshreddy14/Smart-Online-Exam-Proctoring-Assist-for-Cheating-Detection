from flask import Flask
from flask_pymongo import PyMongo
import os

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://jagadeesh:8074563501@cluster0.ur44l.mongodb.net/examproctordb?retryWrites=true&w=majority"
mongo = PyMongo(app)

def seed_database():
    print("Connecting to MongoDB to seed data...")
    try:
        # Create Student
        if not mongo.db.students.find_one({"Email": "student@test.com"}):
            mongo.db.students.insert_one({
                "Name": "Dummy Student",
                "Email": "student@test.com",
                "Password": "password",
                "Role": "STUDENT"
            })
            print("CREATED: Student (student@test.com / password)")
        else:
            print("EXISTING: Student already exists")

        # Create Admin
        if not mongo.db.students.find_one({"Email": "admin@test.com"}):
            mongo.db.students.insert_one({
                "Name": "Dummy Admin",
                "Email": "admin@test.com",
                "Password": "password",
                "Role": "ADMIN"
            })
            print("CREATED: Admin (admin@test.com / password)")
        else:
            print("EXISTING: Admin already exists")
            
        print("\nCurrent Users in DB:")
        for s in mongo.db.students.find():
             print(f"- {s.get('Name')} ({s.get('Role')}): {s.get('Email')}")
             
    except Exception as e:
        print(f"Error seeding database: {e}")

if __name__ == "__main__":
    with app.app_context():
        seed_database()
