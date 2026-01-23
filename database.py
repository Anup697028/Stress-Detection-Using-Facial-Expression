import sqlite3
import json
import os
from werkzeug.security import generate_password_hash # Import for hashing default admin password

DATABASE = 'database.db'
RESULTS_DB_PATH = "autism_results.json"

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE, -- Added email column
                role TEXT NOT NULL DEFAULT 'user'
            );
        ''')

        # Add email column if it doesn't exist (for existing databases)
        cursor.execute("PRAGMA table_info(users);")
        columns = [col[1] for col in cursor.fetchall()]
        if 'email' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT '';")
            conn.commit()
        
        # Check if default admin user exists, if not, create it
        cursor.execute("SELECT * FROM users WHERE username = ?", ('admin',))
        if cursor.fetchone() is None:
            admin_password_hash = generate_password_hash('123')
            # Insert default admin user with an email
            cursor.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)", ('admin', admin_password_hash, 'admin@example.com', 'admin'))
        
        conn.commit()

def add_user(username, email, password):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            # Default role for new users is 'user'
            cursor.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)", (username, email, password, 'user'))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    except Exception as e:
        print(f"Error adding user: {e}")
        return False

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            # Return user data including role and email
            return {"id": user[0], "username": user[1], "password": user[2], "email": user[3], "role": user[4]}
        return None

def get_all_analysis_results():
    results = []
    if not os.path.exists(RESULTS_DB_PATH):
        return results
    try:
        with open(RESULTS_DB_PATH, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError:
        # Handle empty or malformed JSON file
        results = []
    return results

if __name__ == '__main__':
    init_db()
    print("Database initialized and user table created (if not exists).")
    # Example usage for results:
    # all_results = get_all_analysis_results()
    # print(f"All analysis results: {all_results}")
