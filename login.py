import pyodbc
import subprocess
import webbrowser
import time
import threading
import socket
import requests
from flask import Flask, render_template, request, redirect, flash
import os
import secrets
from cryptography.fernet import Fernet
import json


def get_public_ip():
    return requests.get('https://api.ipify.org').text


public_ip = get_public_ip()
print(public_ip)
app = Flask(__name__)
app.secret_key = 'TFG@2023'

RECAPTCHA_SECRET_KEY = "6LfkXZQrAAAAAKIosm2eIEKwzw6AmblfqY8NDb3D"   # from Google
RECAPTCHA_SITE_KEY = "6LfkXZQrAAAAANLCHFVeHYym1YO0F_6aa9mcbziC"     # for template use

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=localhost,1433;'
    'DATABASE=query_genie;'
    'UID=SA;'
    'PWD=abcd@123456;'
)

# conn = pyodbc.connect(
#    'DRIVER={ODBC Driver 17 for SQL Server};'
#    'SERVER=localhost,1433;'
#    'DATABASE=query_genie;'
#    'UID=genie_user;'
#    'PWD=Genie@1234;'
# )

cursor = conn.cursor()
print("cursror connected")


def get_user_credentials():

    cursor.execute("SELECT username, password FROM dbo.login_credentials")
    result = cursor.fetchall()
    return {row[0]: row[1] for row in result}


@app.route('/')
def login_page():
    # return render_template('login.html')
    return render_template('login.html', site_key=RECAPTCHA_SITE_KEY)


# Generate once and store securely; here hardcoded for demo
fernet_key = b'Sv_cBtT5H5i_fv3sPvRrAe_2z6WRnqbmq-rmfxUyiGQ='  # Fernet.generate_key()
cipher_suite = Fernet(fernet_key)


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    recaptcha_response = request.form.get('g-recaptcha-response')

    # Step 1: Verify reCAPTCHA
    captcha_verify = requests.post(
        "https://www.google.com/recaptcha/api/siteverify",
        data={
            'secret': RECAPTCHA_SECRET_KEY,
            'response': recaptcha_response
        }
    )
    result = captcha_verify.json()
    if not result.get('success'):
        flash("reCAPTCHA verification failed.")
        return redirect('/')

    # Step 2: Validate username/password
    credentials = get_user_credentials()
    print("cred connected")
    if username in credentials and credentials[username] == password:
        flash("Login successful ✅")
        print(f"INSERT INTO dbo.login_tracker (username, loginTime) VALUES (?, GETDATE())", (username,))

        # cursor = database_connection().cursor()
        cursor.execute(
            f"INSERT INTO dbo.login_tracker (username, loginTime) VALUES (?, GETDATE())", (username,))
        # cursor.execute(f"INSERT INTO dbo.login_tracker (username, loginTime) VALUES ({username}, GETDATE())")
        conn.commit()

        token = secrets.token_hex(16)

        data = json.dumps({"username": username, "token": token}).encode()
        encrypted_data = cipher_suite.encrypt(data).decode()

        session_data = {
            "username": username,
            "encrypted_data": encrypted_data
        }

        # Clean up old session file if exists
        if os.path.exists("session_token.json"):
            os.remove("session_token.json")

        with open("session_token.json", "w") as f:
            json.dump(session_data, f)
            print(f'Session token saved for {username}')

        subprocess.Popen(['streamlit', 'run', 'Back_Test_mahindra.py', "--server.port", "8501", "--server.address", "0.0.0.0",
                          "--server.headless", "true"])

        return redirect(f"http://localhost:8501")
    else:
        flash("Invalid credentials ❌")
        return redirect('/')


if __name__ == '__main__':
    # app.run(debug=True, port=5000)
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://127.0.0.1:5000")
        print(f"DataGenie Running on http://127.0.0.1:5000")

    # threading.Thread(target=open_browser).start()
    # print("Flask Server Is Starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
