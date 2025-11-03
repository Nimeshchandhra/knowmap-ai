
import sqlite3
import hashlib
import jwt
import datetime
import time
import json
import pandas as pd
import streamlit as st
from config import DB_FILE, SECRET_KEY

# ---------------- DB + Auth Backend ----------------
def init_db():
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    cu.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            is_admin INTEGER DEFAULT 0,
            display_name TEXT DEFAULT NULL,
            theme TEXT DEFAULT 'light'
        )
    """)
    cu.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp INTEGER,
            inputs_count INTEGER,
            triples_count INTEGER,
            nodes_count INTEGER,
            edges_count INTEGER,
            status TEXT,
            details TEXT,
            processed_items INTEGER DEFAULT 0,
            extraction_accuracy REAL DEFAULT NULL
        )
    """)
    cu.execute("""
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp INTEGER,
            query TEXT,
            top_nodes TEXT,
            rating INTEGER,
            comment TEXT
        )
    """)
    
    # --- NEW: Table for user datasets ---
    cu.execute("""
        CREATE TABLE IF NOT EXISTS user_datasets(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_at INTEGER,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    # --- End of new table ---
    
    # default admin
    try:
        cu.execute("SELECT id FROM users WHERE username='admin'")
        if cu.fetchone() is None:
            hashed = hashlib.sha256("admin".encode()).hexdigest()
            cu.execute("INSERT INTO users (username, password, is_admin, display_name) VALUES (?, ?, 1, ?)", ("admin", hashed, "Administrator"))
    except Exception:
        pass
    con.commit()
    con.close()


def add_user(username, password, is_admin: bool = False):
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cu.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", (username, hashed, 1 if is_admin else 0))
    con.commit()
    con.close()


def verify_user(username, password):
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cu.execute("SELECT id, username, is_admin, display_name, theme FROM users WHERE username=? AND password=?", (username, hashed))
    user = cu.fetchone()
    con.close()
    return user


def create_token(username):
    payload = {"username": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_token(token):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"]) if token else None
        return decoded.get("username") if decoded else None
    except jwt.ExpiredSignatureError:
        st.warning("⚠️ Session expired. Please log in again.")
        return None
    except jwt.InvalidTokenError:
        return None

# ---------------- DB helpers for profiles + feedback ----------------
def get_user_profile(username: str):
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    cu.execute("SELECT username, display_name, is_admin, theme FROM users WHERE username=?", (username,))
    row = cu.fetchone()
    con.close()
    if row:
        return {"username": row[0], "display_name": row[1], "is_admin": bool(row[2]), "theme": row[3]}
    return None


def update_profile(username: str, display_name: str = None, theme: str = None):
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    if display_name is not None:
        cu.execute("UPDATE users SET display_name=? WHERE username=?", (display_name, username))
    if theme is not None:
        cu.execute("UPDATE users SET theme=? WHERE username=?", (theme, username))
    con.commit()
    con.close()


def add_feedback(username: str, query: str, top_nodes: list, rating: int, comment: str = ""):
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    cu.execute("INSERT INTO feedback (username, timestamp, query, top_nodes, rating, comment) VALUES (?, ?, ?, ?, ?, ?)",
               (username, int(time.time()), query, json.dumps(top_nodes), rating, comment))
    con.commit()
    con.close()


def fetch_pipeline_runs(limit=200):
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    cu.execute("SELECT id, username, timestamp, inputs_count, triples_count, nodes_count, edges_count, status, processed_items, extraction_accuracy FROM pipeline_runs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cu.fetchall()
    con.close()
    cols = ["id","username","timestamp","inputs_count","triples_count","nodes_count","edges_count","status","processed_items","extraction_accuracy"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def fetch_feedback(limit=200):
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    cu.execute("SELECT id, username, timestamp, query, top_nodes, rating, comment FROM feedback ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cu.fetchall()
    con.close()
    cols = ["id","username","timestamp","query","top_nodes","rating","comment"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# --- NEW: Functions for managing user datasets ---

def add_dataset(username: str, dataset_name: str, file_path: str):
    """Adds a new dataset record to the database."""
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    timestamp = int(time.time())
    cu.execute(
        "INSERT INTO user_datasets (username, dataset_name, file_path, uploaded_at) VALUES (?, ?, ?, ?)",
        (username, dataset_name, file_path, timestamp)
    )
    con.commit()
    con.close()

def get_datasets(username: str) -> pd.DataFrame:
    """Fetches all datasets for a specific user."""
    con = sqlite3.connect(DB_FILE)
    query = "SELECT id, dataset_name, file_path, uploaded_at FROM user_datasets WHERE username = ? ORDER BY uploaded_at DESC"
    df = pd.read_sql_query(query, con, params=(username,))
    con.close()
    return df

def get_dataset_filepath(dataset_id: int, username: str) -> str:
    """Fetches the file path for a specific dataset, ensuring user owns it."""
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    cu.execute("SELECT file_path FROM user_datasets WHERE id = ? AND username = ?", (dataset_id, username))
    row = cu.fetchone()
    con.close()
    return row[0] if row else None

def delete_dataset(dataset_id: int, username: str):
    """Deletes a dataset record from the database, ensuring user owns it."""
    con = sqlite3.connect(DB_FILE)
    cu = con.cursor()
    # We must check username to ensure a user can't delete another's data
    cu.execute("DELETE FROM user_datasets WHERE id = ? AND username = ?", (dataset_id, username))
    con.commit()
    con.close()
