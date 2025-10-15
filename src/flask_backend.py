# src/flask_backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# ----- MongoDB Setup -----
client = MongoClient("mongodb+srv://areebbana:Areeb.2003@cluster0.87efl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["finance_users"]
users = db["user_accounts"]
transactions = db["user_transactions"]

@app.route('/')
def home():
    return jsonify({"message": "Flask backend running successfully!"})

# ----- Registration -----
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if users.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = generate_password_hash(password)
    users.insert_one({"email": email, "password": hashed_pw})
    return jsonify({"message": "User registered successfully!"}), 201

# ----- Login -----
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = users.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"message": "Login successful!", "email": email}), 200

# ----- Save Transactions -----
@app.route('/save_transactions', methods=['POST'])
def save_transactions():
    data = request.get_json()
    email = data.get("email")
    records = data.get("transactions")

    if not email or not records:
        return jsonify({"error": "Missing email or transaction data"}), 400

    transactions.insert_many([
        {"email": email, **rec} for rec in records
    ])
    return jsonify({"message": "Transactions saved successfully!"}), 201

# ----- Fetch Transactions -----
@app.route('/get_transactions', methods=['POST'])
def get_transactions():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"error": "Missing email"}), 400

    user_data = list(transactions.find({"email": email}, {"_id": 0}))
    return jsonify(user_data), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
