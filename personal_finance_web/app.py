import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")

MONGO_URI = os.getenv("MONGO_URI", "")
client = MongoClient(MONGO_URI) if MONGO_URI else None
db = client.finance_app if client else None

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not db:
            flash("Database not configured. Set MONGO_URI in .env")
            return redirect(url_for("register"))
        if db.users.find_one({"email": email}):
            flash("Email already registered.")
            return redirect(url_for("register"))
        hashed = generate_password_hash(password)
        db.users.insert_one({"email": email, "password": hashed})
        flash("Registration successful. Please log in.")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not db:
            flash("Database not configured. Set MONGO_URI in .env")
            return redirect(url_for("login"))
        user = db.users.find_one({"email": email})
        if user and check_password_hash(user.get("password", ""), password):
            session["user"] = email
            flash("Welcome back!")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
