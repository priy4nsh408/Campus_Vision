from flask import Flask, render_template, request, redirect, url_for, session
from query_data import query_rag
import requests
from datetime import timedelta
import csv
import os

# Load environment variables from a .env file when available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_change_me")

# Session configuration
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.permanent_session_lifetime = timedelta(hours=24)

# -----------------------------
# GOOGLE OAUTH CONFIG
# -----------------------------
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:5000/login/callback")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_PEOPLE_API = "https://people.googleapis.com/v1/people/me?personFields=names,emailAddresses"


import csv
import os

import csv
import os

def load_locations_from_csv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "Campus_Map_Description.csv")
    locations = {}

    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Location Name"].strip()
            locations[name] = {
                "lat": float(row["latitiude"]),
                "lng": float(row["longitude"]),
                "category": row.get("Category", "Campus Location"),
                "description": row.get("Description", "")
            }

    return locations



# -----------------------------
# HOME / CHAT PAGE
# -----------------------------
@app.route("/")
def index():
    if "user" not in session:
        return render_template("login.html")

    history = session.get("chat_history", [])

    return render_template(
        "index.html",
        user=session.get("user"),
        history=history
    )


# -----------------------------
# INTRODUCTION PAGE
# -----------------------------
@app.route("/introduction")
def introduction():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "introduction.html",
        user=session.get("user")
    )


# -----------------------------
# GOOGLE LOGIN
# -----------------------------
@app.route("/login")
def login():
    auth_url = (
        f"{GOOGLE_AUTH_URL}"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=openid email profile"
    )
    return redirect(auth_url)


@app.route("/login/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return "Login failed", 400

    token_data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    token_res = requests.post(GOOGLE_TOKEN_URL, data=token_data).json()
    access_token = token_res.get("access_token")

    headers = {"Authorization": f"Bearer {access_token}"}
    user_info = requests.get(GOOGLE_PEOPLE_API, headers=headers).json()

    session["user"] = {
        "name": user_info["names"][0]["displayName"],
        "email": user_info["emailAddresses"][0]["value"],
    }

    if "chat_history" not in session:
        session["chat_history"] = []

    # 🔴 CHANGE: Redirect to introduction page
    return redirect(url_for("introduction"))


# -----------------------------
# LOGOUT
# -----------------------------
@app.route("/logout")
def logout():
    session.clear()
    return render_template("login.html")


@app.route("/clear-history")
def clear_history():
    if "user" in session:
        session["chat_history"] = []
        session.modified = True
    return redirect(url_for("index"))

@app.route("/map")
def map_page():
    if "user" not in session:
        return redirect(url_for("index"))

    locations = load_locations_from_csv()

    return render_template(
        "map.html",
        locations=locations,
        maps_api_key=os.environ.get("MAPS_API_KEY", "")
    )



@app.route("/api/get-directions", methods=["POST"])
def api_get_directions():
    data = request.json
    origin_name = data.get("origin")
    destination_name = data.get("destination")
    waypoint_names = data.get("waypoints", [])

    locations = load_locations_from_csv()

    if origin_name not in locations or destination_name not in locations:
        return {"success": False, "error": "Invalid locations"}, 400

    origin = locations[origin_name]
    destination = locations[destination_name]

    origin_str = f"{origin['lat']},{origin['lng']}"
    destination_str = f"{destination['lat']},{destination['lng']}"

    waypoints_str = None
    if waypoint_names:
        waypoint_coords = []
        for wp in waypoint_names:
            if wp in locations:
                l = locations[wp]
                waypoint_coords.append(f"{l['lat']},{l['lng']}")
        waypoints_str = "|".join(waypoint_coords)

    params = {
        "origin": origin_str,
        "destination": destination_str,
        "mode": "walking",
        "key": os.environ.get("MAPS_API_KEY", "")
    }

    if waypoints_str:
        params["waypoints"] = waypoints_str

    resp = requests.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        params=params
    ).json()

    if resp["status"] != "OK":
        return {"success": False, "error": resp["status"]}, 500

    route = resp["routes"][0]
    leg = route["legs"][0]

    directions = {
        "origin": {
            "name": origin_name,
            "lat": origin["lat"],
            "lng": origin["lng"]
        },
        "destination": {
            "name": destination_name,
            "lat": destination["lat"],
            "lng": destination["lng"]
        },
        "distance": leg["distance"]["text"],
        "duration": leg["duration"]["text"],
        "polyline": route["overview_polyline"]["points"],
        "steps": [
            {
                "number": i + 1,
                "instruction": step["html_instructions"],
                "distance": step["distance"]["text"],
                "duration": step["duration"]["text"]
            }
            for i, step in enumerate(leg["steps"])
        ]
    }

    return {"success": True, "directions": directions}


# ----------------
# 
# -------------
# ASK QUESTION (CHAT)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        return redirect(url_for("login"))

    question = request.form.get("question", "").strip()
    top_k = 1

    if not question:
        return render_template(
            "index.html",
            error="Please enter a question.",
            user=session.get("user"),
            history=session.get("chat_history", [])
        )

    try:
        answer = query_rag(
            question,
            top_k=top_k,
            debug=False,
            show_context=False,
            return_sources=False,
            prioritize_database=True,
            relevance_threshold=1.0,
            suppress_output=True,
        )
        if isinstance(answer, tuple):
            answer = answer[0]
    except Exception as e:
        answer = f"Error: {str(e)}"

    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append({"role": "user", "content": question})
    session["chat_history"].append({"role": "bot", "content": answer})
    session.modified = True

    return render_template(
        "index.html",
        user=session.get("user"),
        history=session.get("chat_history", [])
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
