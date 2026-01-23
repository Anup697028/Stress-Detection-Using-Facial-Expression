# app.py
from flask import Flask, session, request, redirect, url_for, render_template_string
from flask_session import Session
from datetime import timedelta
import secrets
import logging

app = Flask(__name__)
# IMPORTANT: generate a strong secret key in production (env var or secret manager)
app.config['SECRET_KEY'] = secrets.token_urlsafe(32)

# Use server-side sessions (filesystem for demo). In production use Redis or database.
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session'  # ensure this dir is writable
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=20)  # short lifetime

# Cookie flags to mitigate theft via network/XSS
app.config['SESSION_COOKIE_HTTPONLY'] = True
# Set to True in production when serving over HTTPS
app.config['SESSION_COOKIE_SECURE'] = False  # set True when using HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # 'Strict' or 'Lax' recommended

# Apply server-side session
Session(app)

# Basic in-memory "users" for demo only
USERS = {'alice': 'password123'}

# Simple templates (for demo)
LOGIN_FORM = """
<!doctype html>
<title>Login (demo)</title>
<h2>Login</h2>
<form method="post" action="{{ url_for('login') }}">
  <label>Username: <input name="username"></label><br>
  <label>Password: <input name="password" type="password"></label><br>
  <input type="submit" value="Login">
</form>
"""

HOME_PAGE = """
<!doctype html>
<title>Home</title>
<h2>Welcome {{ username }}</h2>
<p>Your session token (server-side) is: {{ token }}</p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
"""

@app.before_request
def log_request_info():
    app.logger.debug('Request path=%s, remote=%s, UA=%s', request.path, request.remote_addr, request.headers.get('User-Agent'))

def regenerate_session(user_id):
    """
    Rotate session: clear old session and create a fresh one with a new token.
    This makes previously stolen session IDs useless after login.
    """
    session.clear()
    session.permanent = True
    session['user_id'] = user_id
    # server-side token to bind session (not sent directly to client)
    session['session_token'] = secrets.token_urlsafe(32)

@app.route('/', methods=['GET'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(HOME_PAGE, username=session['user_id'], token=session.get('session_token'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template_string(LOGIN_FORM)
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    # Very simple auth for demo only. Replace with secure password hashing in production.
    if USERS.get(username) == password:
        regenerate_session(username)
        app.logger.info('User %s logged in, new session token created', username)
        return redirect(url_for('index'))
    app.logger.warning('Failed login attempt for user=%s from %s', username, request.remote_addr)
    return 'Invalid credentials', 401

@app.route('/logout')
def logout():
    session.clear()
    app.logger.info('User logged out; session cleared')
    return 'Logged out'

if __name__ == '__main__':
    # Set logger level to show messages in console
    logging.basicConfig(level=logging.DEBUG)
    # In dev use HTTP; in prod ALWAYS serve via HTTPS and set SESSION_COOKIE_SECURE = True
    app.run(host='127.0.0.1', port=5000, debug=True)
