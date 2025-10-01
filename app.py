"""
Flask Face Recognition Gate (InsightFace + GymMaster Integration)
-----------------------------------------------------------------
Single-file Flask app that:
1) Uses InsightFace (ArcFace) for high-accuracy face recognition.
2) Compares live camera frames against embeddings stored in MySQL (`member.enc`).
3) If a best match is found within threshold, logs in to GymMaster using memberid and opens the gate.
4) Provides a Retake/Compare page where a logged-in user (email+password) can fetch their profile photo
   from GymMaster, capture a new live photo, and compare both via face recognition.

Environment (.env) expected keys:
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
GYM_API_KEY, GYM_BASE_URL, GYM_LOGIN_URL, GYM_PROFILE_URL, GYM_GATE_URL, CHECKIN_ENABLED
GCS_BUCKET_NAME, GCS_BASE_URL_ASSET (optional; not strictly used here)

Run:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r <(echo "flask\npython-dotenv\nmysql-connector-python\nrequests\ninsightface\nonnxruntime\nopencv-python-headless\nnumpy")
  python flask_face_gate_app.py

Open:
  http://localhost:8080/?doorid=<YOUR_DOOR_ID>    # Gate page (Start Camera enabled when doorid present)
  http://localhost:8080/retake                    # Retake/Compare page
"""

import base64
import io
import json
import os
import pickle
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import cv2
import mysql.connector
import numpy as np
import redis
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for

# -------------------- Config & Globals --------------------
load_dotenv()

APP_PORT = int(os.getenv("PORT", 8080))
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

DB = dict(
    host=os.getenv("DB_HOST", "127.0.0.1"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME", "deepface"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", ""),
)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

GYM_API_KEY = os.getenv("GYM_API_KEY", "")
GYM_BASE_URL = os.getenv("GYM_BASE_URL", "https://ftl.gymmasteronline.com")
GYM_LOGIN_URL = os.getenv("GYM_LOGIN_URL", "https://ftl.gymmasteronline.com/portal/api/v1/login")
GYM_PROFILE_URL = os.getenv("GYM_PROFILE_URL", "https://ftl.gymmasteronline.com/portal/api/v1/member/profile")
# For profile update, we need to use a different endpoint or method
# Based on the env, the profile URL is for GET, we need to find the correct UPDATE endpoint
GYM_GATE_URL = os.getenv("GYM_GATE_URL", "https://ftl.gymmasteronline.com/portal/api/v2/member/kiosk/checkin")
CHECKIN_ENABLED = True

# InsightFace model (ArcFace) setup
# Use onnxruntime (CPU by default) for portability.
# Model will lazy-load on first use to avoid slow start.
_insightface_lock = threading.Lock()
_face_rec_model = None
_face_det = None


@dataclass
class MemberEnc:
    member_pk: int
    gym_member_id: int
    email: Optional[str]
    enc: np.ndarray  # 512-d embedding


# Thresholds for ArcFace cosine similarity
SIM_THRESHOLD_MATCH = 0.38  # lower = stricter (tuned; adjust if needed)
TOP2_MARGIN = 0.06          # best must beat second best by this margin


def load_insightface():
    global _face_rec_model, _face_det
    with _insightface_lock:
        if _face_rec_model is None:
            from insightface.app import FaceAnalysis
            _face_rec_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # arcface_r100 / scrfd
            _face_rec_model.prepare(ctx_id=0, det_size=(640, 640))
            _face_det = _face_rec_model  # detector is part of FaceAnalysis
    return _face_rec_model, _face_det


# -------------------- DB Helpers --------------------

def get_db_conn():
    return mysql.connector.connect(
        host=DB["host"], port=DB["port"], database=DB["database"],
        user=DB["user"], password=DB["password"], autocommit=True
    )


def get_redis_conn():
    """Get Redis connection with fallback handling"""
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=False,  # We'll handle binary data
            socket_connect_timeout=5,
            socket_timeout=5
        )
        # Test connection
        r.ping()
        return r
    except Exception as e:
        print(f"DEBUG: Redis connection failed: {e}")
        return None


def fetch_member_encodings() -> List[MemberEnc]:
    """
    Expected table structure (example):
      member (id BIGINT PK, gym_member_id BIGINT, email VARCHAR, enc LONGTEXT)
    where `enc` stores a JSON array of floats (length 512) or a base64 npy blob.
    """
    conn = None
    cur = None
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # First, check if enc column exists, if not create it
        cur.execute("SHOW COLUMNS FROM member LIKE 'enc'")
        if not cur.fetchone():
            print("DEBUG: Adding enc column to member table")
            cur.execute("ALTER TABLE member ADD COLUMN enc LONGTEXT")
            conn.commit()
        
        cur.execute(
            """
            SELECT id AS member_pk, member_id AS gym_member_id, 
                   CONCAT(first_name, ' ', last_name) AS email, enc
            FROM member
            WHERE enc IS NOT NULL AND enc != ''
            """
        )
        out = []
        for member_pk, gym_id, email, enc_raw in cur.fetchall():
            if enc_raw is None:
                continue
            try:
                # Try JSON first
                vec = np.array(json.loads(enc_raw), dtype=np.float32)
                if vec.ndim == 1:
                    pass
                elif vec.ndim == 2 and vec.shape[0] == 1:
                    vec = vec[0]
                else:
                    vec = vec.flatten()
            except Exception:
                # Try base64 npy
                try:
                    arr = np.load(io.BytesIO(base64.b64decode(enc_raw)))
                    vec = arr.astype(np.float32).flatten()
                except Exception:
                    continue
            if vec.size == 0:
                continue
            # Normalize for cosine similarity
            norm = np.linalg.norm(vec) + 1e-8
            vec = vec / norm
            out.append(MemberEnc(member_pk=member_pk, gym_member_id=gym_id or 0, email=email, enc=vec))
        return out
    except Exception as e:
        print(f"DEBUG: Database error in fetch_member_encodings: {e}")
        return []
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# Cache encodings in memory for faster search
_MEMBER_CACHE: List[MemberEnc] = []

# Throttling system to prevent spam
_LAST_RECOGNITION_TIME = {}  # {member_id: timestamp}
_RECOGNITION_COOLDOWN = 10  # 60 seconds cooldown per user (increased from 30)

# Cache refresh system
_LAST_CACHE_REFRESH = 0  # Timestamp of last cache refresh
_CACHE_REFRESH_INTERVAL = 300  # 5 minutes - refresh cache every 5 minutes

# Redis cache keys
REDIS_MEMBER_CACHE_KEY = "face_gate:member_encodings"
REDIS_PROFILE_CACHE_KEY = "face_gate:profile:{}"  # {} will be replaced with member_id
REDIS_CACHE_TTL = 3600  # 1 hour cache TTL


def get_member_encodings_from_redis() -> Optional[List[MemberEnc]]:
    """Get member encodings from Redis cache"""
    r = get_redis_conn()
    if not r:
        return None
    
    try:
        cached_data = r.get(REDIS_MEMBER_CACHE_KEY)
        if cached_data:
            print("DEBUG: Loading member encodings from Redis cache...")
            members = pickle.loads(cached_data)
            print(f"DEBUG: Loaded {len(members)} member encodings from Redis")
            return members
    except Exception as e:
        print(f"DEBUG: Error loading from Redis: {e}")
    
    return None


def save_member_encodings_to_redis(members: List[MemberEnc]):
    """Save member encodings to Redis cache"""
    r = get_redis_conn()
    if not r:
        return
    
    try:
        print("DEBUG: Saving member encodings to Redis cache...")
        serialized = pickle.dumps(members)
        r.setex(REDIS_MEMBER_CACHE_KEY, REDIS_CACHE_TTL, serialized)
        print(f"DEBUG: Saved {len(members)} member encodings to Redis")
    except Exception as e:
        print(f"DEBUG: Error saving to Redis: {e}")


def ensure_cache_loaded(force_refresh=False):
    global _MEMBER_CACHE, _LAST_CACHE_REFRESH
    import time
    
    current_time = time.time()
    
    # Check if we need to refresh cache (every 5 minutes)
    should_refresh = (
        force_refresh or 
        (current_time - _LAST_CACHE_REFRESH) > _CACHE_REFRESH_INTERVAL
    )
    
    # If force_refresh is True, skip cache and reload from database
    if should_refresh:
        print("DEBUG: Cache refresh needed, reloading from database...")
        _MEMBER_CACHE = []
        invalidate_member_cache()
        _LAST_CACHE_REFRESH = current_time
    
    if not _MEMBER_CACHE:
        # Try Redis first (only if not force refresh)
        if not should_refresh:
            cached_members = get_member_encodings_from_redis()
            if cached_members:
                _MEMBER_CACHE = cached_members
                print(f"DEBUG: Loaded {len(_MEMBER_CACHE)} members from Redis cache")
        
        # If no cache or force refresh, load from database
        if not _MEMBER_CACHE:
            try:
                print("DEBUG: Loading member encodings from database...")
                _MEMBER_CACHE = fetch_member_encodings()
                print(f"DEBUG: Loaded {len(_MEMBER_CACHE)} member encodings from database")
                
                # Save to Redis for next time
                if _MEMBER_CACHE:
                    save_member_encodings_to_redis(_MEMBER_CACHE)
                
                for member in _MEMBER_CACHE:
                    print(f"DEBUG: Member {member.member_pk}: {member.email} (gym_id: {member.gym_member_id})")
            except Exception as e:
                print(f"DEBUG: Error loading member cache: {e}")
                _MEMBER_CACHE = []


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def is_user_throttled(member_id: int) -> bool:
    """Check if user is in cooldown period"""
    import time
    current_time = time.time()
    
    if member_id in _LAST_RECOGNITION_TIME:
        time_since_last = current_time - _LAST_RECOGNITION_TIME[member_id]
        if time_since_last < _RECOGNITION_COOLDOWN:
            remaining_time = _RECOGNITION_COOLDOWN - time_since_last
            print(f"DEBUG: User {member_id} is throttled. {remaining_time:.1f}s remaining")
            return True
    
    return False


def get_user_cooldown_remaining(member_id: int) -> int:
    """Get remaining cooldown time in seconds for user"""
    import time
    current_time = time.time()
    
    if member_id in _LAST_RECOGNITION_TIME:
        time_since_last = current_time - _LAST_RECOGNITION_TIME[member_id]
        if time_since_last < _RECOGNITION_COOLDOWN:
            remaining_time = _RECOGNITION_COOLDOWN - time_since_last
            return int(remaining_time)
    
    return 0


def mark_user_recognized(member_id: int):
    """Mark user as recognized to start cooldown"""
    import time
    _LAST_RECOGNITION_TIME[member_id] = time.time()
    print(f"DEBUG: User {member_id} marked as recognized. Cooldown started.")


def find_best_match(query_vec: np.ndarray) -> Tuple[Optional[MemberEnc], float, float]:
    """Return (best_member, best_score, second_best_score)."""
    try:
        ensure_cache_loaded()
        if not _MEMBER_CACHE:
            print("DEBUG: No members in cache")
            return None, 0.0, 0.0
        # Query already normalized
        sims = [(m, float(np.dot(query_vec, m.enc))) for m in _MEMBER_CACHE]
        sims.sort(key=lambda x: x[1], reverse=True)
        best = sims[0]
        second = sims[1] if len(sims) > 1 else (None, 0.0)
        print(f"DEBUG: Best match: {best[0].email if best[0] else 'None'} (score: {best[1]:.4f})")
        return best[0], best[1], second[1]
    except Exception as e:
        print(f"DEBUG: Error in find_best_match: {e}")
        return None, 0.0, 0.0


# -------------------- GymMaster API Helpers --------------------

def gym_login_with_memberid(memberid: int) -> Optional[str]:
    payload = {"api_key": GYM_API_KEY, "memberid": memberid}
    try:
        r = requests.post(GYM_LOGIN_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("error") is None:
            return data["result"]["token"]
    except Exception:
        return None
    return None


def gym_login_with_email(email: str, password: str) -> Optional[Dict]:
    payload = {"api_key": GYM_API_KEY, "email": email, "password": password}
    try:
        r = requests.post(GYM_LOGIN_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("error") is None:
            return data["result"]  # contains token, expires, memberid
    except Exception:
        return None
    return None


def gym_open_gate(token: str, doorid: int) -> Optional[Dict]:
    payload = {"api_key": GYM_API_KEY, "doorid": doorid, "token": token}
    
    # Try different possible endpoints for gate opening
    possible_endpoints = [
        GYM_GATE_URL,  # Original from env
        f"{GYM_BASE_URL}/portal/api/v2/member/kiosk/checkin",  # Try without 'e' at the end
        f"{GYM_BASE_URL}/portal/api/v1/member/kiosk/checkin",   # Try v1 instead of v2
        f"{GYM_BASE_URL}/portal/api/v1/gate/open",              # Try different path
        f"{GYM_BASE_URL}/portal/api/v2/gate/open",              # Try v2 gate
        f"{GYM_BASE_URL}/portal/api/v1/member/checkin",        # Try member checkin
        f"{GYM_BASE_URL}/portal/api/v2/member/checkin"         # Try v2 member checkin
    ]
    
    for i, endpoint in enumerate(possible_endpoints):
        try:
            print(f"DEBUG: Trying gate endpoint {i+1}: {endpoint}")
            print(f"DEBUG: Payload: {payload}")
            
            r = requests.post(endpoint, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
            
            print(f"DEBUG: Gate API response: {data}")
            
            if data.get("error") is None:
                print(f"DEBUG: Gate opened successfully with endpoint {i+1}")
                return data["result"]["response"]
            else:
                print(f"DEBUG: Gate API error: {data.get('error')}")
                
        except Exception as e:
            print(f"DEBUG: Endpoint {i+1} failed: {e}")
            continue
    
    print(f"DEBUG: All gate endpoints failed")
    return None


def get_profile_from_redis(member_id: int) -> Optional[Dict]:
    """Get profile data from Redis cache"""
    r = get_redis_conn()
    if not r:
        return None
    
    try:
        cache_key = REDIS_PROFILE_CACHE_KEY.format(member_id)
        cached_data = r.get(cache_key)
        if cached_data:
            print(f"DEBUG: Loading profile for member {member_id} from Redis cache...")
            profile = json.loads(cached_data)
            print(f"DEBUG: Loaded profile from Redis: {profile.get('fullname', 'Unknown')}")
            return profile
    except Exception as e:
        print(f"DEBUG: Error loading profile from Redis: {e}")
    
    return None


def save_profile_to_redis(member_id: int, profile: Dict):
    """Save profile data to Redis cache"""
    r = get_redis_conn()
    if not r:
        return
    
    try:
        cache_key = REDIS_PROFILE_CACHE_KEY.format(member_id)
        serialized = json.dumps(profile)
        r.setex(cache_key, REDIS_CACHE_TTL, serialized)
        print(f"DEBUG: Saved profile for member {member_id} to Redis cache")
    except Exception as e:
        print(f"DEBUG: Error saving profile to Redis: {e}")


def invalidate_member_cache():
    """Invalidate member encodings cache (both Redis and memory)"""
    global _MEMBER_CACHE
    _MEMBER_CACHE = []
    
    r = get_redis_conn()
    if r:
        try:
            r.delete(REDIS_MEMBER_CACHE_KEY)
            print("DEBUG: Invalidated member encodings cache in Redis")
        except Exception as e:
            print(f"DEBUG: Error invalidating member cache in Redis: {e}")


def invalidate_profile_cache(member_id: int):
    """Invalidate profile cache for specific member"""
    r = get_redis_conn()
    if r:
        try:
            cache_key = REDIS_PROFILE_CACHE_KEY.format(member_id)
            r.delete(cache_key)
            print(f"DEBUG: Invalidated profile cache for member {member_id}")
        except Exception as e:
            print(f"DEBUG: Error invalidating profile cache: {e}")


def clear_all_cache():
    """Clear all Redis cache"""
    r = get_redis_conn()
    if r:
        try:
            # Get all keys with our prefix
            keys = r.keys("face_gate:*")
            if keys:
                r.delete(*keys)
                print(f"DEBUG: Cleared {len(keys)} cache entries from Redis")
        except Exception as e:
            print(f"DEBUG: Error clearing Redis cache: {e}")


def gym_get_profile(token: str) -> Optional[Dict]:
    # Extract member_id from token if possible for caching
    member_id = None
    try:
        import jwt
        # Try to decode token to get member_id (if it's a JWT)
        decoded = jwt.decode(token, options={"verify_signature": False})
        member_id = decoded.get('id')
    except:
        pass
    
    # Try Redis cache first if we have member_id
    if member_id:
        cached_profile = get_profile_from_redis(member_id)
        if cached_profile:
            return cached_profile
    
    # Use GET request with query parameters as specified
    endpoint = f"{GYM_BASE_URL}/portal/api/v1/member/profile"
    params = {
        "token": token,
        "api_key": GYM_API_KEY
    }
    
    print(f"DEBUG: Using GET request to: {endpoint}")  # Debug logging
    print(f"DEBUG: Query parameters: {params}")  # Debug logging
    
    try:
        r = requests.get(endpoint, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        print(f"DEBUG: gym_get_profile response: {data}")  # Debug logging
        
        if data.get("error") is None:
            result = data.get("result")
            print(f"DEBUG: profile result type: {type(result)}")  # Debug logging
            print(f"DEBUG: profile result value: {result}")  # Debug logging
            
            # Check if result is a dictionary with profile data
            if isinstance(result, dict) and "memberphoto" in result:
                print(f"DEBUG: Found profile data with memberphoto: {result.get('memberphoto')}")  # Debug logging
                # Save to Redis cache
                if member_id:
                    save_profile_to_redis(member_id, result)
                return result
            elif isinstance(result, dict) and len(result) > 1:  # Has multiple fields, likely profile data
                print(f"DEBUG: Found profile data with keys: {list(result.keys())}")  # Debug logging
                # Save to Redis cache
                if member_id:
                    save_profile_to_redis(member_id, result)
                return result
            else:
                print(f"DEBUG: Result is not profile data: {result}")  # Debug logging
                return None
        else:
            print(f"DEBUG: API error: {data.get('error')}")  # Debug logging
            return None
    except Exception as e:
        print(f"DEBUG: Exception in gym_get_profile: {e}")  # Debug logging
    return None


# -------------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY


LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FTL Face Gate - Login</title>
    <style>
        body { 
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 400px;
        }
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .login-header h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .login-header p {
            color: #666;
            margin: 0;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
        }
        .form-group input:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
        }
        .login-btn {
            width: 100%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .login-btn:hover {
            background: #0056b3;
        }
        .login-btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .back-btn {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background: #6c757d;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            text-align: center;
            transition: background-color 0.2s;
        }
        .back-btn:hover {
            background: #545b62;
        }
        .error-message {
            color: #dc3545;
            font-size: 14px;
            margin-top: 10px;
            text-align: center;
        }
        .success-message {
            color: #28a745;
            font-size: 14px;
            margin-top: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <h1>FTL Face Gate</h1>
            <p>Login to Register Face</p>
        </div>
        
        <form id="loginForm">
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" id="email" name="email" required>
            </div>
            
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            
            <button type="submit" class="login-btn" id="loginBtn">Login</button>
        </form>
        
        <div id="message"></div>
        
        <a href="/" class="back-btn">← Back to Face Recognition</a>
    </div>

    <script>
        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const loginBtn = document.getElementById('loginBtn');
            const messageDiv = document.getElementById('message');
            
            loginBtn.disabled = true;
            loginBtn.textContent = 'Logging in...';
            messageDiv.innerHTML = '';
            
            try {
                const response = await fetch('/api/retake_login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        email: email,
                        password: password
                    })
                });
                
                const result = await response.json();
                
                if (result.ok) {
                    messageDiv.innerHTML = '<div class="success-message">Login successful! Redirecting...</div>';
                    setTimeout(() => {
                        window.location.href = '/retake';
                    }, 1500);
                } else {
                    messageDiv.innerHTML = '<div class="error-message">Login failed: ' + (result.error || 'Invalid credentials') + '</div>';
                }
            } catch (error) {
                messageDiv.innerHTML = '<div class="error-message">Login error: ' + error.message + '</div>';
            } finally {
                loginBtn.disabled = false;
                loginBtn.textContent = 'Login';
            }
        });
    </script>
</body>
</html>
"""

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FTL Face Gate</title>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
  <style>
    body { 
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; 
      margin: 0; 
      padding: 20px;
      background: #f5f5f5;
      min-height: 100vh;
    }
    .row { display: flex; gap: 24px; align-items: flex-start; }
    video, canvas, img { 
      width: 420px; 
      height: 315px; 
      background: #111; 
      border-radius: 12px; 
      object-fit: cover; 
      max-width: 100%;
    }
    button { 
      padding: 10px 16px; 
      border-radius: 10px; 
      border: 1px solid #ddd; 
      cursor: pointer; 
      transition: all 0.3s;
    }
    button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .ok { color: #0a7; }
    .warn { color: #b70; }
    .err { color: #c31; }
    .pill { padding: 2px 8px; border: 1px solid #999; border-radius: 999px; font-size: 12px; }
    .card { border: 1px solid #eee; padding: 16px; border-radius: 12px; background: white; }
    .muted { color: #777; }
    .field { display: inline-flex; gap: 8px; align-items: center; }
    .stack { display: grid; gap: 12px; }
    
    /* Responsive Design */
    @media (max-width: 768px) {
      body { margin: 10px; padding: 10px; }
      .row { flex-direction: column; gap: 16px; }
      #cameraContainer { 
        height: 400px !important; 
        min-height: 400px !important;
      }
      video, canvas, img { 
        width: 100%; 
        height: 100%;
        object-fit: contain !important;
      }
      button { padding: 10px 16px; font-size: 14px; }
    }
    
    @media (max-width: 480px) {
      body { margin: 5px; padding: 5px; }
      #cameraContainer { 
        height: 350px !important; 
        min-height: 350px !important;
      }
      video, canvas, img { 
        width: 100%; 
        height: 100%;
        object-fit: contain !important;
      }
      button { padding: 8px 12px; font-size: 12px; }
    }
    
    /* Fullscreen Styles */
    .fullscreen {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      z-index: 9999 !important;
      background: black !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      margin: 0 !important;
    }
    
    .fullscreen #video {
      width: 100vw !important;
      height: 100vh !important;
      max-width: none !important;
      min-height: 100vh !important;
      border-radius: 0 !important;
      object-fit: cover !important;
    }
    
    .fullscreen #overlay {
      width: 100vw !important;
      height: 100vh !important;
      border-radius: 0 !important;
    }
    
    .fullscreen #cameraStatus {
      top: 20px !important;
      left: 20px !important;
      font-size: 16px !important;
      padding: 10px 15px !important;
    }
    
    .fullscreen #fullscreenBtn {
      top: 20px !important;
      right: 20px !important;
      font-size: 16px !important;
      padding: 12px 16px !important;
    }
    
    .fullscreen .detection-info {
      position: absolute !important;
      bottom: 20px !important;
      left: 50% !important;
      transform: translateX(-50%) !important;
      background: rgba(0,0,0,0.8) !important;
      color: white !important;
      border-radius: 10px !important;
      padding: 15px 25px !important;
      max-width: 90vw !important;
    }
    
    /* Fullscreen Exit Button */
    .fullscreen-exit {
      position: absolute !important;
      top: 20px !important;
      right: 80px !important;
      background: rgba(220, 53, 69, 0.8) !important;
      color: white !important;
      border: none !important;
      padding: 12px 16px !important;
      border-radius: 8px !important;
      cursor: pointer !important;
      font-size: 16px !important;
      z-index: 10000 !important;
    }
    
    .fullscreen-exit:hover {
      background: rgba(220, 53, 69, 1) !important;
    }
  </style>
</head>
<body>
    <div style="max-width: 800px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); padding: 32px;">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px;">
        <div>
          <h1 style="margin: 0 0 8px 0; font-size: 28px; font-weight: 700; color: #333;">FTL Face Gate</h1>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #666; font-size: 14px;">Door ID:</span>
            <span id="doorid" class="pill" style="background: #495057; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;">19456</span>
          </div>
        </div>
        <div style="color: #999; font-size: 12px;">Open preview in new tab</div>
      </div>
      
        <!-- Camera Display Area -->
        <div id="cameraContainer" style="position: relative; width: 100%; height: 450px; background: #f8f9fa; border-radius: 12px; margin-bottom: 24px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
          <video id="video" autoplay playsinline muted style="width: 100%; height: 100%; background: #f8f9fa; border-radius: 12px; object-fit: contain; border: none; transform: scaleX(-1); display: none;"></video>
          <canvas id="overlay" style="position: absolute; top: 0; left: 0; pointer-events: none; border-radius: 12px; width: 100%; height: 100%; z-index: 10; background: transparent;"></canvas>
          
          <!-- Camera Placeholder -->
          <div id="cameraPlaceholder" style="display: flex; flex-direction: column; align-items: center; justify-content: center; color: #6c757d;">
            <div style="width: 48px; height: 36px; border: 2px solid #6c757d; border-radius: 8px; position: relative; margin-bottom: 12px;">
              <div style="width: 16px; height: 16px; border: 2px solid #6c757d; border-radius: 50%; position: absolute; top: 8px; left: 14px;"></div>
            </div>
            <div style="font-size: 16px; font-weight: 500; color: #6c757d;">Camera Inactive</div>
          </div>
          
          <!-- Status Badge -->
          <div id="cameraStatus" style="position: absolute; top: 12px; left: 12px; background: #495057; color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;">
            Offline
          </div>
          
          <!-- Fullscreen Button -->
          <button id="fullscreenBtn" style="position: absolute; top: 12px; right: 12px; background: #495057; color: white; border: none; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; width: 32px; height: 32px;">
            <i class="fas fa-expand" style="font-size: 12px;"></i>
          </button>
        </div>
      
      <!-- Control Buttons -->
      <div style="margin-bottom: 24px; display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;">
        <button id="btnStart" disabled style="padding: 12px 20px; border: none; border-radius: 8px; background: #007bff; color: white; cursor: pointer; font-weight: 500; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 4px rgba(0,123,255,0.3);">
          <i class="fas fa-camera" style="font-size: 14px;"></i>
          <span>Start Camera</span>
        </button>
        <button id="btnStop" disabled style="padding: 12px 20px; border: none; border-radius: 8px; background: #dc3545; color: white; cursor: pointer; font-weight: 500; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 4px rgba(220,53,69,0.3);">
          <i class="fas fa-stop" style="font-size: 14px;"></i>
          <span>Stop Camera</span>
        </button>
        <button id="btnRetake" onclick="location.href='/login'" style="padding: 12px 20px; border: none; border-radius: 8px; background: #28a745; color: white; cursor: pointer; font-weight: 500; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 4px rgba(40,167,69,0.3);">
          <i class="fas fa-user-plus" style="font-size: 14px;"></i>
          <span>Register Face</span>
        </button>
        <button id="btnDebug" onclick="debugCamera()" style="padding: 12px 20px; border: none; border-radius: 8px; background: #ffc107; color: #212529; cursor: pointer; font-weight: 500; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 4px rgba(255,193,7,0.3);">
          <i class="fas fa-cog" style="font-size: 14px;"></i>
          <span>Debug Camera</span>
        </button>
        <button id="btnToggleOverlay" onclick="toggleOverlay()" style="padding: 12px 20px; border: none; border-radius: 8px; background: #20c997; color: white; cursor: pointer; font-weight: 500; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 4px rgba(32,201,151,0.3);">
          <i class="fas fa-eye" style="font-size: 14px;"></i>
          <span>Show Overlay</span>
        </button>
      </div>
      
      <!-- Status Cards -->
      <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px;">
        <!-- Face Detection Status -->
        <div id="detectionResult" style="padding: 16px; border-radius: 12px; background: #f8f9fa; border: 1px solid #e9ecef; display: flex; align-items: center; gap: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
          <div style="width: 32px; height: 32px; border-radius: 50%; background: #dc3545; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px;">
            <i class="fas fa-times"></i>
          </div>
          <div id="detectedName" style="font-size: 16px; font-weight: 500; color: #333;">No face detected</div>
        </div>
        
        <!-- Camera Status -->
        <div id="cameraStatusCard" style="padding: 16px; border-radius: 12px; background: #f8f9fa; border: 1px solid #e9ecef; display: flex; align-items: center; gap: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
          <div id="cameraStatusIcon" style="width: 32px; height: 32px; border-radius: 50%; background: #dc3545; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px;">
            <i class="fas fa-times"></i>
          </div>
          <div id="detectionStatus" style="font-size: 16px; font-weight: 500; color: #333;">Camera inactive</div>
        </div>
        
        <!-- Countdown Timer -->
        <div id="countdownTimer" style="display: none; padding: 16px; border-radius: 12px; background: #fff3cd; border: 1px solid #ffeaa7; display: flex; align-items: center; gap: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: all 0.3s ease; opacity: 0; transform: translateY(10px);">
          <div style="width: 32px; height: 32px; border-radius: 50%; background: #ffc107; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px;">
            <i class="fas fa-clock"></i>
          </div>
          <div style="flex: 1;">
            <div style="font-size: 14px; font-weight: 500; color: #856404;">Next scan available in: <span id="countdownSeconds" style="font-weight: bold; color: #d63384;">0</span> seconds</div>
          </div>
        </div>
      </div>
      
      <div style="color: #999; font-size: 14px; text-align: center; margin-top: 16px;">
        Face recognition runs automatically when camera is active.
      </div>
    </div>
  <script>
    const q = new URLSearchParams(location.search);
    const doorid = q.get('doorid');
    document.getElementById('doorid').textContent = doorid || '—';

    const video = document.getElementById('video');
    const overlay = document.getElementById('overlay');
    const cameraStatus = document.getElementById('cameraStatus');
    const detectedName = document.getElementById('detectedName');
    const detectionStatus = document.getElementById('detectionStatus');
    
    // Create canvas for image capture
    let canvas = null;

    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');

    let stream = null;
      let recognitionInterval = null;
      let lastRecognitionTime = 0;
      let isProcessing = false; // Flag to prevent multiple simultaneous requests
      let countdownInterval = null; // For countdown timer
      let remainingCooldown = 0; // Remaining cooldown time in seconds
      let isFullscreen = false; // Fullscreen state
      let lastSuccessfulRecognitionTime = 0; // Track when user was last successfully recognized

    function setLog(obj) {
      console.log(typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2));
    }
    
    function drawBoundingBox(x, y, width, height, color = '#00FF00', label = '') {
      if (!overlay) return;
      
      const ctx = overlay.getContext('2d');
      const videoWidth = video.videoWidth || 640;
      const videoHeight = video.videoHeight || 480;
      
      // Get actual display dimensions
      const displayWidth = video.clientWidth;
      const displayHeight = video.clientHeight;
      
      // Set canvas size to match video display size (responsive)
      overlay.width = displayWidth;
      overlay.height = displayHeight;
      overlay.style.width = displayWidth + 'px';
      overlay.style.height = displayHeight + 'px';
      
      // Clear canvas with transparent background
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      
      if (x && y && width && height) {
        // Scale coordinates from video resolution to display resolution
        const scaleX = displayWidth / videoWidth;
        const scaleY = displayHeight / videoHeight;
        
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        // Mirror the coordinates to match the mirrored video
        const mirroredX = displayWidth - scaledX - scaledWidth;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, 3 * Math.min(scaleX, scaleY)); // Responsive line width
        ctx.strokeRect(mirroredX, scaledY, scaledWidth, scaledHeight);
        
        if (label) {
          ctx.fillStyle = color;
          const fontSize = Math.max(12, 16 * Math.min(scaleX, scaleY)); // Responsive font size
          ctx.font = `bold ${fontSize}px Arial`;
          ctx.fillText(label, mirroredX, Math.max(scaledY - 10, 20));
        }
        
        // Debug logging for mobile
        console.log('Bounding box debug:', {
          original: { x, y, width, height },
          videoSize: { videoWidth, videoHeight },
          displaySize: { displayWidth, displayHeight },
          scale: { scaleX, scaleY },
          scaled: { scaledX, scaledY, scaledWidth, scaledHeight },
          mirrored: { mirroredX }
        });
      }
    }

    function clearBoundingBox() {
      if (!overlay) return;
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function syncOverlayWithVideo() {
      if (!overlay || !video) return;
      
      const displayWidth = video.clientWidth;
      const displayHeight = video.clientHeight;
      
      // Ensure overlay matches video display size exactly
      overlay.width = displayWidth;
      overlay.height = displayHeight;
      overlay.style.width = displayWidth + 'px';
      overlay.style.height = displayHeight + 'px';
      
      console.log('Overlay synced with video:', displayWidth, 'x', displayHeight);
    }

    function setButtons(running) {
      btnStart.disabled = running || !doorid;
      btnStop.disabled = !running;
    }

    function updateDetectionDisplay(name, status, confidence = null) {
      const detectionResult = document.getElementById('detectionResult');
      const statusIcon = detectionResult.querySelector('i');
      const statusCircle = detectionResult.querySelector('div');
      
      // Update camera status card
      const cameraStatusCard = document.getElementById('cameraStatusCard');
      const cameraStatusIcon = document.getElementById('cameraStatusIcon');
      const cameraStatusIconElement = cameraStatusIcon.querySelector('i');
      
      detectedName.textContent = name || 'No face detected';
      detectionStatus.textContent = status;
      
      // Check if it's a success message (gate opened, recognized, etc.)
      const isSuccess = status.toLowerCase().includes('success') || 
                       status.toLowerCase().includes('opened') || 
                       status.toLowerCase().includes('recognized') ||
                       status.toLowerCase().includes('gate opened');
      
      if (isSuccess) {
        // Success state - green checkmark for both cards
        detectedName.style.color = '#28a745';
        statusIcon.className = 'fas fa-check';
        statusCircle.style.background = '#28a745';
        
        // Update camera status card to success
        cameraStatusIconElement.className = 'fas fa-check';
        cameraStatusIcon.style.background = '#28a745';
        detectionStatus.style.color = '#28a745';
      } else if (name && confidence) {
        // Face detected with confidence
        detectedName.style.color = confidence > 0.6 ? '#28a745' : '#ffc107';
        detectionStatus.textContent = `${status} (${Math.round(confidence * 100)}% confidence)`;
        
        // Update status icon and color
        statusIcon.className = confidence > 0.6 ? 'fas fa-check' : 'fas fa-exclamation-triangle';
        statusCircle.style.background = confidence > 0.6 ? '#28a745' : '#ffc107';
        
        // Update camera status card
        cameraStatusIconElement.className = confidence > 0.6 ? 'fas fa-check' : 'fas fa-exclamation-triangle';
        cameraStatusIcon.style.background = confidence > 0.6 ? '#28a745' : '#ffc107';
        detectionStatus.style.color = confidence > 0.6 ? '#28a745' : '#ffc107';
      } else if (name) {
        // Face detected but no confidence
        detectedName.style.color = '#ffc107';
        statusIcon.className = 'fas fa-exclamation-triangle';
        statusCircle.style.background = '#ffc107';
        
        // Update camera status card
        cameraStatusIconElement.className = 'fas fa-exclamation-triangle';
        cameraStatusIcon.style.background = '#ffc107';
        detectionStatus.style.color = '#ffc107';
      } else {
        // No face detected or error
        detectedName.style.color = '#333';
        statusIcon.className = 'fas fa-times';
        statusCircle.style.background = '#dc3545';
        
        // Update camera status card
        cameraStatusIconElement.className = 'fas fa-times';
        cameraStatusIcon.style.background = '#dc3545';
        detectionStatus.style.color = '#333';
      }
    }

    function showCountdownTimer(seconds) {
      const countdownTimer = document.getElementById('countdownTimer');
      const countdownSeconds = document.getElementById('countdownSeconds');
      
      console.log('showCountdownTimer called with:', seconds);
      
      if (seconds > 0) {
        // Clear any existing interval
        if (countdownInterval) {
          clearInterval(countdownInterval);
        }
        
        // Show timer with smooth transition
        countdownTimer.style.display = 'flex';
        countdownTimer.style.opacity = '0';
        countdownTimer.style.transform = 'translateY(10px)';
        
        // Animate in
        setTimeout(() => {
          countdownTimer.style.transition = 'all 0.3s ease';
          countdownTimer.style.opacity = '1';
          countdownTimer.style.transform = 'translateY(0)';
        }, 10);
        
        countdownSeconds.textContent = seconds;
        remainingCooldown = seconds;
        
        // Start countdown
        countdownInterval = setInterval(() => {
          remainingCooldown--;
          countdownSeconds.textContent = remainingCooldown;
          
          console.log('Countdown:', remainingCooldown);
          
          if (remainingCooldown <= 0) {
            hideCountdownTimer();
          }
        }, 1000);
      } else {
        hideCountdownTimer();
      }
    }

      function hideCountdownTimer() {
        const countdownTimer = document.getElementById('countdownTimer');
        
        console.log('hideCountdownTimer called');
        
        // Animate out
        countdownTimer.style.transition = 'all 0.3s ease';
        countdownTimer.style.opacity = '0';
        countdownTimer.style.transform = 'translateY(-10px)';
        
        setTimeout(() => {
          countdownTimer.style.display = 'none';
        }, 300);
        
        if (countdownInterval) {
          clearInterval(countdownInterval);
          countdownInterval = null;
        }
        remainingCooldown = 0;
      }

      function toggleFullscreen() {
        const cameraContainer = document.getElementById('cameraContainer');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        
        if (!isFullscreen) {
          // Enter fullscreen
          cameraContainer.classList.add('fullscreen');
          fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i> Exit';
          
          // Add exit button
          const exitBtn = document.createElement('button');
          exitBtn.id = 'fullscreenExitBtn';
          exitBtn.className = 'fullscreen-exit';
          exitBtn.innerHTML = '<i class="fas fa-times"></i> Exit Fullscreen';
          exitBtn.onclick = toggleFullscreen;
          cameraContainer.appendChild(exitBtn);
          
          isFullscreen = true;
          console.log('Entered fullscreen mode');
        } else {
          // Exit fullscreen
          cameraContainer.classList.remove('fullscreen');
          fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i> Fullscreen';
          
          // Remove exit button
          const exitBtn = document.getElementById('fullscreenExitBtn');
          if (exitBtn) {
            exitBtn.remove();
          }
          
          isFullscreen = false;
          console.log('Exited fullscreen mode');
        }
      }

      function handleFullscreenChange() {
        // Handle browser fullscreen API
        if (!document.fullscreenElement && isFullscreen) {
          toggleFullscreen();
        }
      }

    async function startCam() {
      try {
        console.log('Requesting camera access...');
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: 'user', 
            width: { ideal: 1280, max: 1920 }, 
            height: { ideal: 720, max: 1080 },
            aspectRatio: { ideal: 16/9 }
          } 
        });
        console.log('Camera stream obtained:', stream);
        
        video.srcObject = stream;
        console.log('Video srcObject set');
        
        // Wait for video to load
        video.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
          
          // Show video and hide placeholder
          video.style.display = 'block';
          document.getElementById('cameraPlaceholder').style.display = 'none';
          
          // Update status
          cameraStatus.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; color: #28a745;"></i><span>Online</span>';
          cameraStatus.style.background = 'rgba(40, 167, 69, 0.9)';
          
          // Sync overlay with video dimensions
          syncOverlayWithVideo();
        };
        
        video.oncanplay = () => {
          console.log('Video can play');
          cameraStatus.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; color: #28a745;"></i><span>Streaming</span>';
          cameraStatus.style.background = 'rgba(40, 167, 69, 0.9)';
        };
        
        video.onerror = (e) => {
          console.error('Video error:', e);
          setLog('Video error: ' + e.message);
        };
        
        // Force play the video
        video.play().catch(e => {
          console.error('Video play error:', e);
        });
        
        // Additional check to ensure video is playing
        setTimeout(() => {
          if (video.paused) {
            console.log('Video is paused, attempting to play...');
            video.play().catch(e => console.error('Retry play error:', e));
          }
          if (video.videoWidth === 0) {
            console.log('Video dimensions are 0, stream may not be working');
            cameraStatus.textContent = 'Camera error - no video';
            cameraStatus.style.background = 'rgba(200,0,0,0.8)';
          }
        }, 1000);
        
        // Create canvas for image capture
        canvas = document.createElement('canvas');
        canvas.width = 1280;
        canvas.height = 720;
        console.log('Canvas created');
        
        setButtons(true);
        setLog('Camera started.');
        updateDetectionDisplay(null, 'Camera active - detecting faces...');
        
        // Hide any existing countdown timer when camera starts
        hideCountdownTimer();
        
        // Start automatic face recognition every 4 seconds
        recognitionInterval = setInterval(performRecognition, 4000);
        console.log('Recognition interval started');
      } catch (e) {
        console.error('Camera error:', e);
        setLog('Camera error: ' + e.message);
        updateDetectionDisplay(null, 'Camera error: ' + e.message);
        cameraStatus.textContent = 'Camera error: ' + e.message;
        cameraStatus.style.background = 'rgba(200,0,0,0.8)';
      }
    }

    function stopCam() {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
        video.srcObject = null;
        stream = null;
      }
      if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
      }
      hideCountdownTimer(); // Hide countdown when camera stops
      setButtons(false);
      setLog('Camera stopped.');
      updateDetectionDisplay(null, 'Camera inactive');
      
      // Show placeholder and hide video
      video.style.display = 'none';
      document.getElementById('cameraPlaceholder').style.display = 'flex';
      
      // Update status
      cameraStatus.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; color: #dc3545;"></i><span>Offline</span>';
      cameraStatus.style.background = 'rgba(108, 117, 125, 0.9)';
    }

    async function performRecognition() {
      if (!stream || !doorid || !canvas || isProcessing) return;
      
      const now = Date.now();
      if (now - lastRecognitionTime < 3000) return; // Increased throttle to 3 seconds
      lastRecognitionTime = now;
      isProcessing = true; // Set flag to prevent multiple requests
      
      try {
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        
        updateDetectionDisplay(null, 'Recognizing...');
        
      const r = await fetch('/api/recognize_open_gate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ doorid: parseInt(doorid), image_b64: dataUrl.split(',')[1] })
        });
        
        let j;
        try {
          j = await r.json();
        } catch (e) {
          console.error('JSON parse error:', e);
          updateDetectionDisplay(null, 'Server error: Invalid response');
          return;
        }
        
        if (j.ok && j.matched && j.candidate) {
          const name = j.candidate.email || `Member ${j.candidate.gym_member_id}`;
          const confidence = j.best_score;
          
          // Immediately show the name and draw bounding box
          updateDetectionDisplay(name, 'Face recognized', confidence);
          
          // Draw bounding box for recognized face
          if (j.bbox && j.bbox.length === 4) {
            const [x1, y1, x2, y2] = j.bbox;
            drawBoundingBox(x1, y1, x2-x1, y2-y1, '#4CAF50', name);
          } else {
            // Draw a default box if no bbox
            drawBoundingBox(100, 100, 200, 200, '#4CAF50', name);
          }
          
          // Check if user is throttled
          if (j.gate && j.gate.throttled) {
            console.log('User is throttled, showing cooldown timer');
            updateDetectionDisplay(name, 'User in cooldown period', confidence);
            
            // Get exact remaining time from server
            const serverCooldownSeconds = j.gate.cooldown_remaining || 10;
            console.log('Server cooldown seconds:', serverCooldownSeconds);
            
            // Calculate time elapsed since last successful recognition
            const currentTime = Date.now();
            
            // If we don't have lastSuccessfulRecognitionTime, use server time directly
            let actualRemainingTime;
            if (lastSuccessfulRecognitionTime === 0) {
              // First time or no previous successful recognition, use server time
              actualRemainingTime = serverCooldownSeconds;
              console.log('No previous successful recognition, using server time:', actualRemainingTime);
            } else {
              const timeSinceLastSuccessfulRecognition = Math.floor((currentTime - lastSuccessfulRecognitionTime) / 1000);
              actualRemainingTime = Math.max(0, serverCooldownSeconds - timeSinceLastSuccessfulRecognition);
              console.log('Time since last successful recognition:', timeSinceLastSuccessfulRecognition, 'seconds');
            }
            
            console.log('Server cooldown:', serverCooldownSeconds, 'seconds');
            console.log('Actual remaining time:', actualRemainingTime, 'seconds');
            
            // Show countdown timer with corrected time
            if (actualRemainingTime > 0) {
              showCountdownTimer(actualRemainingTime);
            } else {
              hideCountdownTimer();
            }
            
            // Change bounding box color to yellow for throttled user
            if (j.bbox && j.bbox.length === 4) {
              const [x1, y1, x2, y2] = j.bbox;
              drawBoundingBox(x1, y1, x2-x1, y2-y1, '#FFC107', 'Cooldown');
            } else {
              drawBoundingBox(100, 100, 200, 200, '#FFC107', 'Cooldown');
            }
          } else if (j.gate && !j.gate.error) {
            // Auto-open gate if matched and not throttled
            console.log('Gate opened successfully, hiding cooldown timer');
            updateDetectionDisplay(name, 'Gate opened successfully!', confidence);
            hideCountdownTimer(); // Hide timer if gate opened successfully
            
            // Record successful recognition time for cooldown calculation
            lastSuccessfulRecognitionTime = Date.now();
            console.log('Successful recognition recorded at:', new Date(lastSuccessfulRecognitionTime));
          }
        } else if (j.ok && !j.matched) {
          updateDetectionDisplay('Unknown person', 'Face detected but not recognized');
          // Draw red bounding box for unknown face
          if (j.bbox && j.bbox.length === 4) {
            const [x1, y1, x2, y2] = j.bbox;
            drawBoundingBox(x1, y1, x2-x1, y2-y1, '#FF9800', 'Unknown');
          } else {
            drawBoundingBox(100, 100, 200, 200, '#FF9800', 'Unknown');
          }
        } else {
          updateDetectionDisplay(null, j.error || 'No face detected');
          clearBoundingBox();
        }
      } catch (e) {
        updateDetectionDisplay(null, 'Recognition error: ' + e.message);
        setLog('Recognition error: ' + e.message);
      } finally {
        isProcessing = false; // Reset flag when done
      }
    }

    btnStart.onclick = () => {
      console.log('Start camera button clicked');
      startCam();
    };
    btnStop.onclick = () => {
      console.log('Stop camera button clicked');
      stopCam();
    };
    
    // Fullscreen functionality
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    if (fullscreenBtn) {
      fullscreenBtn.onclick = toggleFullscreen;
    }
    
    // Listen for fullscreen changes
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);
    
    // Handle window resize for mobile responsiveness
    window.addEventListener('resize', () => {
      if (overlay && video) {
        // Sync overlay with video when window resizes
        syncOverlayWithVideo();
      }
    });
    
    // Handle orientation change for mobile
    window.addEventListener('orientationchange', () => {
      setTimeout(() => {
        if (overlay && video) {
          // Sync overlay with video after orientation change
          syncOverlayWithVideo();
        }
      }, 100);
    });
    
    function debugCamera() {
      console.log('=== CAMERA DEBUG INFO ===');
      console.log('Video element:', video);
      console.log('Video srcObject:', video.srcObject);
      console.log('Video paused:', video.paused);
      console.log('Video readyState:', video.readyState);
      console.log('Video videoWidth:', video.videoWidth);
      console.log('Video videoHeight:', video.videoHeight);
      console.log('Stream:', stream);
      console.log('Stream active:', stream ? stream.active : 'No stream');
      console.log('Canvas:', canvas);
      console.log('Overlay:', overlay);
      console.log('Recognition interval:', recognitionInterval);
      console.log('Door ID:', doorid);
      
      // Test if we can access camera
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          console.log('Camera access test: SUCCESS');
          console.log('Test stream:', stream);
          stream.getTracks().forEach(track => track.stop());
        })
        .catch(err => {
          console.log('Camera access test: FAILED');
          console.log('Error:', err);
        });
    }
    
    function toggleOverlay() {
      const btn = document.getElementById('btnToggleOverlay');
      if (overlay.style.display === 'none') {
        overlay.style.display = 'block';
        btn.textContent = 'Hide Overlay';
        btn.style.background = '#6f42c1';
      } else {
        overlay.style.display = 'none';
        btn.textContent = 'Show Overlay';
        btn.style.background = '#28a745';
      }
    }

    setButtons(false);
    if (doorid) {
      btnStart.disabled = false;
    }
  </script>
</body>
</html>
"""


RETAKE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FTL Retake & Compare</title>
  <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body { 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
      background: #f8f9fa;
      min-height: 100vh;
      padding: 20px;
    }
    
    .header {
      text-align: center;
      margin-bottom: 40px;
    }
    
    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      color: #6f42c1;
      margin-bottom: 10px;
    }
    
    .header p {
      color: #6c757d;
      font-size: 1.1rem;
    }
    
    .container {
      max-width: 1400px;
      margin: 0 auto;
    }
    
    .row { 
      display: flex; 
      gap: 30px; 
      align-items: flex-start; 
      flex-wrap: wrap;
      justify-content: center;
    }
    
    .card { 
      background: white;
      border: 1px solid #e9ecef; 
      padding: 30px; 
      border-radius: 16px; 
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      min-width: 350px;
      max-width: 400px;
    }
    
    .card-header {
      display: flex;
      align-items: center;
      margin-bottom: 25px;
      padding-bottom: 15px;
      border-bottom: 2px solid #f8f9fa;
    }
    
    .card-header h3 {
      font-size: 1.3rem;
      font-weight: 600;
      color: #495057;
      margin-left: 10px;
    }
    
    .card-header i {
      font-size: 1.5rem;
      color: #6f42c1;
    }
    
    .profile-field {
      display: flex;
      align-items: center;
      margin-bottom: 20px;
      padding: 12px;
      background: #f8f9fa;
      border-radius: 10px;
      border: 1px solid #e9ecef;
    }
    
    .profile-field i {
      font-size: 1.2rem;
      color: #6f42c1;
      margin-right: 15px;
      width: 20px;
    }
    
    .profile-field input {
      border: none;
      background: transparent;
      font-size: 1rem;
      color: #495057;
      width: 100%;
      outline: none;
    }
    
    .logout-btn {
      width: 100%;
      padding: 12px 20px;
      background: #dc3545;
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: background-color 0.3s;
    }
    
    .logout-btn:hover {
      background: #c82333;
    }
    
    .status-message {
      margin-top: 15px;
      padding: 10px;
      background: #d4edda;
      color: #155724;
      border-radius: 8px;
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .photo-container {
      width: 100%;
      height: 300px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: white;
      margin-bottom: 20px;
      position: relative;
      overflow: hidden;
    }
    
    .photo-container img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 12px;
    }
    
    .photo-placeholder {
      text-align: center;
    }
    
    .photo-placeholder i {
      font-size: 4rem;
      margin-bottom: 15px;
      opacity: 0.8;
    }
    
    .photo-placeholder p {
      font-size: 1.1rem;
      font-weight: 500;
    }
    
    .photo-info {
      background: #e3f2fd;
      color: #1976d2;
      padding: 12px;
      border-radius: 8px;
      font-size: 0.9rem;
      text-align: center;
      margin-top: 15px;
    }
    
    .camera-container {
      width: 100%;
      height: 300px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: white;
      margin-bottom: 20px;
      position: relative;
      overflow: hidden;
    }
    
    .camera-container video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 12px;
    }
    
    .camera-placeholder {
      text-align: center;
    }
    
    .camera-placeholder i {
      font-size: 4rem;
      margin-bottom: 15px;
      opacity: 0.8;
    }
    
    .camera-placeholder p {
      font-size: 1.1rem;
      font-weight: 500;
    }
    
    .btn-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    
    .btn {
      padding: 12px 20px;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.3s;
    }
    
    .btn-start {
      background: white;
      color: #495057;
      border: 2px solid #e9ecef;
    }
    
    .btn-start:hover {
      background: #f8f9fa;
      border-color: #6f42c1;
    }
    
    .btn-capture {
      background: white;
      color: #495057;
      border: 2px solid #e9ecef;
    }
    
    .btn-capture:hover {
      background: #f8f9fa;
      border-color: #6f42c1;
    }
    
    .btn-capture:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    
    .btn-update {
      background: #28a745;
      color: white;
      border: none;
    }
    
    .btn-update:hover {
      background: #218838;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
    }
    
    .btn-update:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    
    .btn-reset {
      background: #ffc107;
      color: #212529;
      border: none;
    }
    
    .btn-reset:hover {
      background: #e0a800;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(255, 193, 7, 0.4);
    }
    
    .btn-reset:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    
    .btn-register {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
    }
    
    .btn-register:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    canvas { display: none; }
    
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
  <h1>Profile & Face Registration</h1>
      <p>Secure biometric authentication and identity verification system</p>
    </div>
    
  <div class="row">
      <div class="card">
        <div class="card-header">
          <i class="fas fa-user"></i>
      <h3>User Profile</h3>
        </div>
        <div id="profileInfo">
        <div id="loadingProfile">Loading profile...</div>
      </div>
        <button id="btnLogout" class="logout-btn" onclick="logout()">
          <i class="fas fa-arrow-right"></i>
          Logout
        </button>
        <div id="statusMessage" class="status-message hidden">
          <i class="fas fa-check"></i>
          <span>Profile photo loaded successfully</span>
      </div>
        <pre id="log" style="display: none;"></pre>
    </div>
      
      <div class="card">
        <div class="card-header">
          <i class="fas fa-camera"></i>
      <h3>Profile Photo</h3>
    </div>
        <div class="photo-container" id="photoContainer">
          <img id="profile" alt="profile" style="display: none;" />
          <div class="photo-placeholder" id="photoPlaceholder">
            <i class="fas fa-camera"></i>
            <p>Profile photo will be displayed here</p>
          </div>
        </div>
        <div class="photo-info">
          This photo is used as reference for face recognition matching
        </div>
      </div>
      
      <div class="card">
        <div class="card-header">
          <i class="fas fa-user-check"></i>
      <h3>Capture & Compare</h3>
        </div>
        <div class="camera-container" id="cameraContainer">
          <video id="video" autoplay playsinline muted style="display: none; transform: scaleX(-1);"></video>
          <img id="capturedImage" alt="captured" style="display: none; width: 100%; height: 100%; object-fit: cover; border-radius: 12px;" />
          <div class="camera-placeholder" id="cameraPlaceholder">
            <i class="fas fa-camera"></i>
          </div>
        </div>
        <div class="btn-group">
          <button id="btnStart" class="btn btn-start">
            <i class="fas fa-play"></i>
            Start Camera
          </button>
          <button id="btnSnap" class="btn btn-capture" disabled>
            <i class="fas fa-user-check"></i>
            Capture & Compare
          </button>
          <button id="btnRegister" class="btn btn-register">
            <i class="fas fa-user-plus"></i>
            Register Face Recognition
          </button>
        </div>
        <canvas id="canvas" width="640" height="480"></canvas>
        <pre id="out" style="display: none;"></pre>
      </div>
    </div>
  </div>
  
  <!-- Modal for Face Registration -->
  <div id="registerModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; padding: 24px; border-radius: 12px; max-width: 800px; width: 90%; max-height: 90vh; overflow-y: auto;">
      <h2 style="text-align: center; margin-bottom: 24px; color: #333;">Register Face Recognition</h2>
      
      <!-- Camera Section -->
      <div style="text-align: center; margin: 20px 0;">
        <div style="position: relative; width: 100%; max-width: 500px; height: 300px; margin: 0 auto; background: #111; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
          <video id="registerVideo" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); display: none; border-radius: 12px; position: absolute; top: 0; left: 0; z-index: 2; background: #000;"></video>
          <img id="capturedImage" alt="captured" style="display: none; width: 100%; height: 100%; object-fit: cover; border-radius: 12px; position: absolute; top: 0; left: 0; z-index: 3;" />
          <div id="cameraPlaceholder" style="width: 100%; height: 100%; background: #f8f9fa; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #6c757d; position: absolute; top: 0; left: 0; border-radius: 12px;">
            <i class="fas fa-camera" style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;"></i>
            <div style="font-size: 18px; font-weight: 500;">Camera Inactive</div>
          </div>
        </div>
      </div>
      
      <style>
        @media (max-width: 768px) {
          #registerVideo, #capturedImage {
            object-fit: contain !important;
          }
        }
        @media (max-width: 480px) {
          #registerVideo, #capturedImage {
            object-fit: contain !important;
          }
        }
      </style>
      
      <!-- Control Buttons -->
      <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin: 20px 0;">
        <button id="btnStartRegister" style="padding: 12px 24px; background: #2196F3; color: white; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 8px;">
          <i class="fas fa-play"></i>
          Start Camera
        </button>
        <button id="btnCapturePhoto" disabled style="padding: 12px 24px; background: #28a745; color: white; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 8px;">
          <i class="fas fa-camera"></i>
          Capture Photo
        </button>
        <button id="btnUpdatePhoto" disabled style="padding: 12px 24px; background: #17a2b8; color: white; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 8px;">
          <i class="fas fa-upload"></i>
          Update to GymMaster
        </button>
        <button id="btnResetPhoto" disabled style="padding: 12px 24px; background: #ffc107; color: #212529; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 8px;">
          <i class="fas fa-redo"></i>
          Reset Photo
        </button>
        <button id="btnBurstCapture" disabled style="padding: 12px 24px; background: #FF9800; color: white; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 8px;">
          <i class="fas fa-bolt"></i>
          Burst Capture (5s)
        </button>
        <button id="btnCloseRegister" style="padding: 12px 24px; background: #dc3545; color: white; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 8px;">
          <i class="fas fa-times"></i>
          Close
        </button>
      </div>
      
      <!-- Progress and Status -->
      <div id="registerProgress" style="margin: 16px 0; font-size: 14px; color: #666; text-align: center; min-height: 20px;"></div>
      <canvas id="registerCanvas" width="640" height="480" style="display: none;"></canvas>
    </div>
  </div>
  
  <script>
    const log = document.getElementById('log');
    const out = document.getElementById('out');
    const img = document.getElementById('profile');

    function setLog(x){ log.textContent = typeof x==='string'? x : JSON.stringify(x,null,2); }
    function setOut(x){ out.textContent = typeof x==='string'? x : JSON.stringify(x,null,2); }

    let token=null; let stream=null; let registerStream=null; let burstInterval=null;

    // Auto-load profile when page loads
    async function loadProfile() {
      const profileInfo = document.getElementById('profileInfo');
      const loadingProfile = document.getElementById('loadingProfile');
      
      try {
        if (loadingProfile) {
        loadingProfile.textContent = 'Loading profile...';
        }
        
        // Check if we have session data from login
        const r = await fetch('/api/get_profile', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });
        
        const j = await r.json();
        setLog(j);
        
        if (j.ok && j.profile) {
          token = j.token;
          const profile = j.profile;
          
          // Display profile info with styled fields
          profileInfo.innerHTML = `
            <div class="profile-field">
              <i class="fas fa-user"></i>
              <input type="text" value="${profile.fullname || 'N/A'}" readonly>
            </div>
            <div class="profile-field">
              <i class="fas fa-envelope"></i>
              <input type="text" value="${profile.email || 'N/A'}" readonly>
            </div>
            <div class="profile-field">
              <i class="fas fa-phone"></i>
              <input type="text" value="${profile.phonecell || 'N/A'}" readonly>
            </div>
            <div class="profile-field">
              <i class="fas fa-id-card"></i>
              <input type="text" value="${profile.id || 'N/A'}" readonly>
            </div>
          `;
          
          // Load profile photo
          const photoContainer = document.getElementById('photoContainer');
          const photoPlaceholder = document.getElementById('photoPlaceholder');
          const statusMessage = document.getElementById('statusMessage');
          
          if (profile.memberphoto) {
            setLog('Loading profile photo: ' + profile.memberphoto);
            img.src = profile.memberphoto;
            img.onerror = () => {
              setLog('Error loading profile photo: ' + profile.memberphoto);
              photoPlaceholder.style.display = 'block';
              img.style.display = 'none';
            };
            img.onload = () => {
              setLog('Profile photo loaded successfully');
              img.style.display = 'block';
              photoPlaceholder.style.display = 'none';
              statusMessage.classList.remove('hidden');
            };
          } else {
            setLog('No profile photo found');
            photoPlaceholder.style.display = 'block';
            img.style.display = 'none';
          }
        } else {
          if (profileInfo) {
          profileInfo.innerHTML = '<div style="color: red;">Not logged in. Please go back to login page.</div>';
          }
        }
      } catch (e) {
        if (profileInfo) {
        profileInfo.innerHTML = '<div style="color: red;">Error loading profile: ' + e.message + '</div>';
        }
        if (typeof setLog === 'function') {
        setLog('Error: ' + e.message);
        }
      }
    }

    // Load profile when page loads
    loadProfile();

    // Logout function
    async function logout() {
      try {
        const r = await fetch('/api/logout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });
        
        const j = await r.json();
        setLog('Logout: ' + JSON.stringify(j));
        
        if (j.ok) {
          // Clear local variables
          token = null;
          
          // Redirect to login page
          window.location.href = '/login';
        } else {
          alert('Logout failed: ' + (j.error || 'Unknown error'));
        }
      } catch (e) {
        setLog('Logout error: ' + e.message);
        alert('Logout error: ' + e.message);
      }
    }

    // Add event handler for Start button on retake page
    document.getElementById('btnStart').onclick = async () => {
      try {
        setOut('Starting camera...');
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 } 
        });
        
        const video = document.getElementById('video');
        const cameraContainer = document.getElementById('cameraContainer');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        
        video.srcObject = stream;
        video.style.display = 'block';
        cameraPlaceholder.style.display = 'none';
        
        document.getElementById('btnStart').disabled = true;
        document.getElementById('btnSnap').disabled = false;
        document.getElementById('btnCapturePhoto').disabled = false;
        setOut('Camera started. Click "Capture Photo" to take a photo.');
      } catch (e) {
        setOut('Camera error: ' + e.message);
      }
    };

    // Add event handler for Capture & Compare button
    document.getElementById('btnSnap').onclick = async () => {
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas');
      
      if (!video.srcObject) {
        setOut('Please start camera first');
        return;
      }
      
      try {
        setOut('Capturing photo...');
        
        // Capture current frame
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        
        setOut('Comparing with profile photo...');
        
        // Send to compare API
        const r = await fetch('/api/compare_with_profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_b64: dataUrl.split(',')[1] })
        });
        
        const j = await r.json();
        setOut(j);
        
        if (j.ok) {
          const similarity = j.similarity;
          if (similarity > 0.6) {
            setOut(`Match! Similarity: ${similarity} (High confidence)`);
            // SweetAlert for successful match
            Swal.fire({
              title: '✅ Face Match!',
              text: `Tingkat kesamaan: ${(similarity * 100).toFixed(1)}% (Tinggi)`,
              icon: 'success',
              confirmButtonText: 'OK',
              confirmButtonColor: '#4CAF50',
              timer: 3000,
              timerProgressBar: true
            });
          } else if (similarity > 0.4) {
            setOut(`Possible match. Similarity: ${similarity} (Medium confidence)`);
            // SweetAlert for possible match
            Swal.fire({
              title: '⚠️ Kemungkinan Match',
              text: `Tingkat kesamaan: ${(similarity * 100).toFixed(1)}% (Sedang)`,
              icon: 'warning',
              confirmButtonText: 'OK',
              confirmButtonColor: '#FF9800'
            });
          } else {
            setOut(`No match. Similarity: ${similarity} (Low confidence)`);
            // SweetAlert for no match
            Swal.fire({
              title: '❌ Tidak Match',
              text: `Tingkat kesamaan: ${(similarity * 100).toFixed(1)}% (Rendah) - Beda orang`,
              icon: 'error',
              confirmButtonText: 'OK',
              confirmButtonColor: '#f44336'
            });
          }
        } else {
          setOut('Comparison failed: ' + (j.error || 'Unknown error'));
          // SweetAlert for error
          Swal.fire({
            title: '❌ Error',
            text: 'Gagal membandingkan foto: ' + (j.error || 'Unknown error'),
            icon: 'error',
            confirmButtonText: 'OK',
            confirmButtonColor: '#f44336'
          });
        }
      } catch (e) {
        setOut('Capture error: ' + e.message);
        // SweetAlert for capture error
        Swal.fire({
          title: '❌ Error',
          text: 'Gagal mengambil foto: ' + e.message,
          icon: 'error',
          confirmButtonText: 'OK',
          confirmButtonColor: '#f44336'
        });
      }
    };

    // Add event handler for Capture Photo button
    document.getElementById('btnCapturePhoto').onclick = async () => {
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas');
      const capturedImage = document.getElementById('capturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      if (!video.srcObject) {
        setOut('Please start camera first');
        return;
      }
      
      try {
        setOut('Capturing photo...');
        
        // Capture current frame
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        
        // Show captured image
        capturedImage.src = dataUrl;
        capturedImage.style.display = 'block';
        video.style.display = 'none';
        cameraPlaceholder.style.display = 'none';
        
        // Update button states
        document.getElementById('btnCapturePhoto').disabled = true;
        document.getElementById('btnUpdatePhoto').disabled = false;
        document.getElementById('btnResetPhoto').disabled = false;
        
        setOut('Photo captured! Review the result and click "Update to GymMaster" if satisfied, or "Reset Photo" to retake.');
        
        // Show success notification
        Swal.fire({
          title: '📸 Photo Captured!',
          text: 'Review your photo. If satisfied, click "Update to GymMaster" to upload.',
          icon: 'success',
          confirmButtonText: 'OK',
          confirmButtonColor: '#28a745',
          timer: 3000,
          timerProgressBar: true
        });
        
      } catch (e) {
        setOut('Capture error: ' + e.message);
        Swal.fire({
          title: '❌ Error',
          text: 'Failed to capture photo: ' + e.message,
          icon: 'error',
          confirmButtonText: 'OK',
          confirmButtonColor: '#dc3545'
        });
      }
    };

    // Add event handler for Reset Photo button
    document.getElementById('btnResetPhoto').onclick = () => {
      const video = document.getElementById('video');
      const capturedImage = document.getElementById('capturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      // Reset to camera view
      video.style.display = 'block';
      capturedImage.style.display = 'none';
      cameraPlaceholder.style.display = 'none';
      
      // Update button states
      document.getElementById('btnCapturePhoto').disabled = false;
      document.getElementById('btnUpdatePhoto').disabled = true;
      document.getElementById('btnResetPhoto').disabled = true;
      
      setOut('Photo reset. You can now capture a new photo.');
    };

    // Add event handler for Update Profile Photo button
    document.getElementById('btnUpdatePhoto').onclick = async () => {
      const capturedImage = document.getElementById('capturedImage');
      
      if (!capturedImage.src || capturedImage.style.display === 'none') {
        setOut('Please capture a photo first');
        return;
      }
      
      try {
        // Use the captured image data
        const dataUrl = capturedImage.src;
        
        // Show confirmation dialog
        const result = await Swal.fire({
          title: '⚠️ Update Profile Photo?',
          text: 'Are you sure you want to update your profile photo in GymMaster? This action cannot be undone.',
          icon: 'warning',
          showCancelButton: true,
          confirmButtonColor: '#28a745',
          cancelButtonColor: '#dc3545',
          confirmButtonText: 'Yes, Update Photo',
          cancelButtonText: 'Cancel',
          reverseButtons: true
        });
        
        if (result.isConfirmed) {
          setOut('Updating profile photo...');
          
          // Send to update API
          const r = await fetch('/api/update_profile_photo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: dataUrl.split(',')[1] })
          });
          
          const j = await r.json();
          setOut(j);
          
          if (j.ok) {
            // Success notification
            Swal.fire({
              title: '✅ Success!',
              text: 'Profile photo updated successfully in GymMaster',
              icon: 'success',
              confirmButtonText: 'OK',
              confirmButtonColor: '#28a745',
              timer: 3000,
              timerProgressBar: true
            });
            
            // Update User Profile section with success message
            const profileInfo = document.getElementById('profileInfo');
            const statusMessage = document.getElementById('statusMessage');
            
            // Show success message in User Profile section
            statusMessage.innerHTML = '<i class="fas fa-check"></i><span>Foto telah di update ke gymmaster</span>';
            statusMessage.classList.remove('hidden');
            statusMessage.style.background = '#d4edda';
            statusMessage.style.color = '#155724';
            
            // Reload profile to show updated photo
            setTimeout(() => {
              loadProfile();
            }, 1000);
          } else {
            // Error notification
            Swal.fire({
              title: '❌ Update Failed',
              text: 'Failed to update profile photo: ' + (j.error || 'Unknown error'),
              icon: 'error',
              confirmButtonText: 'OK',
              confirmButtonColor: '#dc3545'
            });
          }
        }
      } catch (e) {
        setOut('Update error: ' + e.message);
        Swal.fire({
          title: '❌ Error',
          text: 'Failed to update profile photo: ' + e.message,
          icon: 'error',
          confirmButtonText: 'OK',
          confirmButtonColor: '#dc3545'
        });
      }
    };

    // Face Registration Modal Controls
    document.getElementById('btnRegister').onclick = () => {
      console.log('Opening register modal...');
      document.getElementById('registerModal').style.display = 'block';
      
      // Debug: Check if elements exist
      const video = document.getElementById('registerVideo');
      const placeholder = document.getElementById('cameraPlaceholder');
      console.log('Video element:', video);
      console.log('Placeholder element:', placeholder);
      
      // Reset modal state
      if (video) {
        video.style.display = 'none';
        video.srcObject = null;
      }
      if (placeholder) {
        placeholder.style.display = 'flex';
      }
      
      // Reset button states
      document.getElementById('btnCapturePhoto').disabled = true;
      document.getElementById('btnUpdatePhoto').disabled = true;
      document.getElementById('btnResetPhoto').disabled = true;
      document.getElementById('btnBurstCapture').disabled = true;
    };

    document.getElementById('btnCloseRegister').onclick = () => {
      if (registerStream) {
        registerStream.getTracks().forEach(track => track.stop());
        registerStream = null;
      }
      if (burstInterval) {
        clearInterval(burstInterval);
        burstInterval = null;
      }
      
      // Reset modal state
      const registerVideo = document.getElementById('registerVideo');
      const capturedImage = document.getElementById('capturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      registerVideo.style.display = 'none';
      capturedImage.style.display = 'none';
      cameraPlaceholder.style.display = 'flex';
      
      // Reset button states
      document.getElementById('btnCapturePhoto').disabled = true;
      document.getElementById('btnUpdatePhoto').disabled = true;
      document.getElementById('btnResetPhoto').disabled = true;
      document.getElementById('btnBurstCapture').disabled = true;
      
      document.getElementById('registerProgress').textContent = '';
      document.getElementById('registerModal').style.display = 'none';
    };

    document.getElementById('btnStartRegister').onclick = async () => {
      try {
        console.log('Starting camera...');
        document.getElementById('registerProgress').textContent = 'Starting camera...';
        
        // Stop any existing stream first
        if (registerStream) {
          registerStream.getTracks().forEach(track => track.stop());
          registerStream = null;
        }
        
        // Request camera with simple constraints
        registerStream = await navigator.mediaDevices.getUserMedia({ 
          video: true
        });
        
        console.log('Camera stream obtained:', registerStream);
        
        const registerVideo = document.getElementById('registerVideo');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        
        console.log('Video element:', registerVideo);
        console.log('Placeholder element:', cameraPlaceholder);
        
        // Set video source
        registerVideo.srcObject = registerStream;
        
        // Add event listeners for video events
        registerVideo.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          registerVideo.style.display = 'block';
          registerVideo.style.visibility = 'visible';
          registerVideo.style.opacity = '1';
          cameraPlaceholder.style.display = 'none';
        };
        
        registerVideo.oncanplay = () => {
          console.log('Video can play');
          registerVideo.style.display = 'block';
          registerVideo.style.visibility = 'visible';
          registerVideo.style.opacity = '1';
          cameraPlaceholder.style.display = 'none';
        };
        
        // Force show video immediately
        registerVideo.style.display = 'block';
        registerVideo.style.visibility = 'visible';
        registerVideo.style.opacity = '1';
        cameraPlaceholder.style.display = 'none';
        
        console.log('Video display style:', registerVideo.style.display);
        console.log('Placeholder display style:', cameraPlaceholder.style.display);
        
        // Force play video
        registerVideo.play().then(() => {
          console.log('Video playing successfully');
          document.getElementById('registerProgress').textContent = 'Camera ready. You can now capture photos.';
        }).catch(e => {
          console.error('Video play error:', e);
          document.getElementById('registerProgress').textContent = 'Video play error: ' + e.message;
        });
        
        // Fallback: Force display after 1 second
        setTimeout(() => {
          if (registerVideo.style.display === 'none' || cameraPlaceholder.style.display !== 'none') {
            console.log('Fallback: Forcing video display');
            registerVideo.style.display = 'block';
            registerVideo.style.visibility = 'visible';
            registerVideo.style.opacity = '1';
            cameraPlaceholder.style.display = 'none';
          }
        }, 1000);
        
        document.getElementById('btnCapturePhoto').disabled = false;
        document.getElementById('btnBurstCapture').disabled = false;
        
      } catch (e) {
        console.error('Camera error:', e);
        document.getElementById('registerProgress').textContent = 'Camera error: ' + e.message;
      }
    };

    // Capture Photo button in modal
    document.getElementById('btnCapturePhoto').onclick = async () => {
      const registerVideo = document.getElementById('registerVideo');
      const capturedImage = document.getElementById('capturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      if (!registerStream) {
        document.getElementById('registerProgress').textContent = 'Please start camera first';
        return;
      }
      
      try {
        const registerCanvas = document.getElementById('registerCanvas');
        const ctx = registerCanvas.getContext('2d');
        ctx.drawImage(registerVideo, 0, 0, registerCanvas.width, registerCanvas.height);
        const dataUrl = registerCanvas.toDataURL('image/jpeg', 0.9);
        
        // Show captured image
        capturedImage.src = dataUrl;
        capturedImage.style.display = 'block';
        registerVideo.style.display = 'none';
        
        // Ensure image loads properly
        capturedImage.onload = () => {
          console.log('Captured image loaded successfully');
        };
        
        // Update button states
        document.getElementById('btnCapturePhoto').disabled = true;
        document.getElementById('btnUpdatePhoto').disabled = false;
        document.getElementById('btnResetPhoto').disabled = false;
        
        document.getElementById('registerProgress').textContent = 'Photo captured! Review and click "Update to GymMaster" if satisfied.';
      } catch (e) {
        document.getElementById('registerProgress').textContent = 'Capture error: ' + e.message;
      }
    };

    // Reset Photo button in modal
    document.getElementById('btnResetPhoto').onclick = () => {
      const registerVideo = document.getElementById('registerVideo');
      const capturedImage = document.getElementById('capturedImage');
      
      // Reset to camera view
      registerVideo.style.display = 'block';
      capturedImage.style.display = 'none';
      
      // Update button states
      document.getElementById('btnCapturePhoto').disabled = false;
      document.getElementById('btnUpdatePhoto').disabled = true;
      document.getElementById('btnResetPhoto').disabled = true;
      
      document.getElementById('registerProgress').textContent = 'Photo reset. You can now capture a new photo.';
    };

    // Update Photo button in modal
    document.getElementById('btnUpdatePhoto').onclick = async () => {
      const capturedImage = document.getElementById('capturedImage');
      
      if (!capturedImage.src || capturedImage.style.display === 'none') {
        document.getElementById('registerProgress').textContent = 'Please capture a photo first';
        return;
      }
      
      try {
        const dataUrl = capturedImage.src;
        
        document.getElementById('registerProgress').textContent = 'Updating profile photo...';
        
        const r = await fetch('/api/update_profile_photo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_b64: dataUrl.split(',')[1] })
        });
        
        const j = await r.json();
        
        if (j.ok) {
          document.getElementById('registerProgress').textContent = 'Profile photo updated successfully!';
          document.getElementById('btnUpdatePhoto').disabled = true;
        } else {
          document.getElementById('registerProgress').textContent = 'Update failed: ' + (j.error || 'Unknown error');
        }
      } catch (e) {
        document.getElementById('registerProgress').textContent = 'Update error: ' + e.message;
      }
    };

    document.getElementById('btnBurstCapture').onclick = async () => {
      if (!registerStream) return;
      
      const registerVideo = document.getElementById('registerVideo');
      const registerCanvas = document.getElementById('registerCanvas');
      const progress = document.getElementById('registerProgress');
      
      let capturedFrames = [];
      let countdown = 5;
      
      progress.textContent = `Burst capture starting in ${countdown} seconds...`;
      document.getElementById('btnBurstCapture').disabled = true;
      
      // Countdown
      const countdownInterval = setInterval(() => {
        countdown--;
        progress.textContent = `Burst capture starting in ${countdown} seconds...`;
        if (countdown <= 0) {
          clearInterval(countdownInterval);
          startBurstCapture();
        }
      }, 1000);
      
      function startBurstCapture() {
        progress.textContent = 'Capturing photos... (5 seconds)';
        
        burstInterval = setInterval(() => {
          const ctx = registerCanvas.getContext('2d');
          ctx.drawImage(registerVideo, 0, 0, registerCanvas.width, registerCanvas.height);
          const dataUrl = registerCanvas.toDataURL('image/jpeg', 0.9);
          capturedFrames.push(dataUrl);
          progress.textContent = `Photo ${capturedFrames.length} captured...`;
        }, 200); // Take photo every 200ms
        
        setTimeout(async () => {
          clearInterval(burstInterval);
          burstInterval = null;
          progress.textContent = 'Sending photos for face encoding...';
          
          // Send frames to server for encoding
          const r = await fetch('/api/register_face', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames: capturedFrames })
          });
          const j = await r.json();
          
          if (j.ok) {
            progress.textContent = 'Face recognition registered successfully!';
            document.getElementById('btnBurstCapture').disabled = false;
          } else {
            progress.textContent = 'Error: ' + j.error;
            document.getElementById('btnBurstCapture').disabled = false;
          }
        }, 5000);
      }
    };
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/login")
def login():
    return render_template_string(LOGIN_HTML)

@app.route("/retake")
def retake():
    # Check if user is logged in
    token = session.get('gym_token')
    if not token:
        # Redirect to login page if not logged in
        return redirect(url_for('login'))
    
    return render_template_string(RETAKE_HTML)


# -------------------- Image & Recognition Utils --------------------

def b64_to_bgr(image_b64: str) -> Optional[np.ndarray]:
    try:
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]
        data = base64.b64decode(image_b64)
        im = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(im, cv2.IMREAD_COLOR)
        return bgr
    except Exception:
        return None


def extract_embedding(bgr: np.ndarray) -> Optional[np.ndarray]:
    model, det = load_insightface()
    faces = det.get(bgr)
    if not faces:
        return None
    # Choose largest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding  # already L2-normalized 512-d
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)

def extract_embedding_with_bbox(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Extract embedding and return bounding box coordinates"""
    model, det = load_insightface()
    faces = det.get(bgr)
    if not faces:
        return None, None
    # Choose largest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding  # already L2-normalized 512-d
    if emb is None:
        return None, None
    
    # Convert bbox to integer coordinates
    bbox = face.bbox
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return np.asarray(emb, dtype=np.float32), (x1, y1, x2, y2)


# -------------------- API Endpoints --------------------
@app.route("/api/recognize_open_gate", methods=["POST"])
def api_recognize_open_gate():
    if not CHECKIN_ENABLED:
        return jsonify({"ok": False, "error": "Check-in disabled by config"}), 400

    data = request.get_json(force=True)
    doorid = data.get("doorid")
    image_b64 = data.get("image_b64")
    if not doorid:
        return jsonify({"ok": False, "error": "doorid is required"}), 400

    bgr = b64_to_bgr(image_b64)
    if bgr is None:
        return jsonify({"ok": False, "error": "Invalid image"}), 400

    emb, bbox = extract_embedding_with_bbox(bgr)
    if emb is None:
        return jsonify({"ok": False, "error": "No face found", "bbox": None}), 200

    best, best_score, second = find_best_match(emb)
    if not best:
        return jsonify({"ok": False, "error": "No enrolled members"}), 200

    margin = best_score - second
    matched = (best_score >= (1.0 - SIM_THRESHOLD_MATCH)) and (margin >= TOP2_MARGIN)

    resp = {
        "ok": True,
        "best_score": round(best_score, 4),
        "second_best": round(second, 4),
        "margin": round(margin, 4),
        "matched": bool(matched),
        "bbox": bbox,  # [x1, y1, x2, y2] coordinates
        "candidate": {
            "member_pk": best.member_pk,
            "gym_member_id": best.gym_member_id,
            "email": best.email,
        },
    }

    if matched and best.gym_member_id:
        print(f"DEBUG: Face matched with member ID: {best.gym_member_id}")
        
        # Check if user is throttled
        if is_user_throttled(best.gym_member_id):
            cooldown_remaining = get_user_cooldown_remaining(best.gym_member_id)
            resp["gate"] = {
                "error": "User in cooldown period", 
                "throttled": True,
                "cooldown_remaining": cooldown_remaining
            }
            return jsonify(resp)
        
        token = gym_login_with_memberid(best.gym_member_id)
        if token:
            print(f"DEBUG: Login successful, token obtained")
            session['gym_token'] = token
            gate = gym_open_gate(token, int(doorid))
            if gate:
                print(f"DEBUG: Gate opened successfully: {gate}")
                resp["gate"] = gate
                # Mark user as recognized to start cooldown
                mark_user_recognized(best.gym_member_id)
            else:
                print(f"DEBUG: Gate opening failed")
                resp["gate"] = {"error": "Gate API failed"}
        else:
            print(f"DEBUG: Login failed for member ID: {best.gym_member_id}")
            resp["gate"] = {"error": "Login API failed"}
    else:
        print(f"DEBUG: Face not matched or no member ID")
        resp["gate"] = {"error": "Face not confidently matched"}

    return jsonify(resp)


@app.route("/api/retake_login", methods=["POST"])
def api_retake_login():
    data = request.get_json(force=True)
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"ok": False, "error": "email and password required"}), 400

    result = gym_login_with_email(email, password)
    if not result:
        return jsonify({"ok": False, "error": "Login failed"})
    token = result.get("token")
    session['gym_token'] = token
    
    # Get profile data using the correct API endpoint
    profile = gym_get_profile(token)
    if profile:
        session['profile_data'] = profile  # Cache profile data in session
    
    # Debug information for profile photo
    debug_info = {}
    if profile:
        debug_info["profile_type"] = type(profile).__name__
        debug_info["profile_data"] = str(profile)[:200] + "..." if len(str(profile)) > 200 else str(profile)
        if isinstance(profile, dict):
            debug_info["profile_photo_url"] = profile.get("memberphoto")
            debug_info["profile_keys"] = list(profile.keys())
        else:
            debug_info["profile_photo_url"] = None
            debug_info["profile_keys"] = []
    else:
        debug_info["profile_type"] = "None"
        debug_info["profile_data"] = "No profile data received"
        debug_info["profile_photo_url"] = None
        debug_info["profile_keys"] = []
    
    return jsonify({
        "ok": True, 
        "token": token, 
        "memberid": result.get("memberid"), 
        "profile": profile,
        "debug": debug_info
    })


@app.route("/api/get_profile", methods=["GET"])
def api_get_profile():
    token = session.get('gym_token')
    if not token:
        return jsonify({"ok": False, "error": "Not logged in. Please login first."}), 401

    # Get profile from session
    profile = session.get('profile_data')
    if not profile:
        # Try to get from API if not in session
        profile = gym_get_profile(token)
        if profile:
            session['profile_data'] = profile

    if not profile:
        return jsonify({"ok": False, "error": "Failed to fetch profile"})

    return jsonify({
        "ok": True,
        "token": token,
        "profile": profile
    })


@app.route("/api/logout", methods=["POST"])
def api_logout():
    # Clear session data
    session.pop('gym_token', None)
    session.pop('profile_data', None)
    
    return jsonify({
        "ok": True,
        "message": "Logged out successfully"
    })


@app.route("/api/compare_with_profile", methods=["POST"])
def api_compare_with_profile():
    token = session.get('gym_token')
    if not token:
        return jsonify({"ok": False, "error": "Not logged in. Please login first."}), 401

    # Try to get profile from session first, then from API if needed
    profile = session.get('profile_data')
    if not profile:
        profile = gym_get_profile(token)
        if profile:
            session['profile_data'] = profile  # Cache for future use
    
    if not profile:
        return jsonify({"ok": False, "error": "Failed to fetch profile"})

    image_b64 = request.json.get("image_b64")
    bgr_live = b64_to_bgr(image_b64)
    if bgr_live is None:
        return jsonify({"ok": False, "error": "Invalid image"}), 400

    # Fetch profile photo and embed
    if not isinstance(profile, dict):
        return jsonify({"ok": False, "error": f"Profile data is not a dictionary, got: {type(profile).__name__}"})
    
    profile_url = profile.get("memberphoto")
    if not profile_url:
        return jsonify({"ok": False, "error": "No profile photo on GymMaster"})

    try:
        r = requests.get(profile_url, timeout=15)
        r.raise_for_status()
        im = np.frombuffer(r.content, np.uint8)
        bgr_prof = cv2.imdecode(im, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"ok": False, "error": "Failed to download profile photo"})

    emb_live = extract_embedding(bgr_live)
    emb_prof = extract_embedding(bgr_prof)

    if emb_live is None:
        return jsonify({"ok": False, "error": "No face in live capture"})
    if emb_prof is None:
        return jsonify({"ok": False, "error": "No face in profile photo"})

    sim = float(np.dot(emb_live, emb_prof) / (np.linalg.norm(emb_live)*np.linalg.norm(emb_prof)+1e-8))
    return jsonify({"ok": True, "similarity": round(sim, 4), "threshold_note": "higher is more similar; ~0.6-0.8 usually same person for ArcFace crops"})


@app.route("/api/cache/clear", methods=["POST"])
def api_clear_cache():
    """Clear all cache (Redis + memory)"""
    try:
        clear_all_cache()
        invalidate_member_cache()
        return jsonify({"ok": True, "message": "All cache cleared successfully"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cache/status", methods=["GET"])
def api_cache_status():
    """Get cache status information"""
    try:
        r = get_redis_conn()
        redis_status = "connected" if r else "disconnected"
        
        cache_info = {
            "redis_status": redis_status,
            "memory_cache_size": len(_MEMBER_CACHE),
            "cache_ttl": REDIS_CACHE_TTL,
            "refresh_interval": _CACHE_REFRESH_INTERVAL,
            "last_refresh": _LAST_CACHE_REFRESH
        }
        
        if r:
            try:
                keys = r.keys("face_gate:*")
                cache_info["redis_keys_count"] = len(keys)
                cache_info["redis_memory_usage"] = r.memory_usage("face_gate:*")
            except:
                cache_info["redis_keys_count"] = 0
                cache_info["redis_memory_usage"] = 0
        
        return jsonify({"ok": True, "cache": cache_info})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cache/refresh", methods=["POST"])
def api_refresh_cache():
    """Manually refresh cache from database"""
    try:
        ensure_cache_loaded(force_refresh=True)
        return jsonify({
            "ok": True, 
            "message": "Cache refreshed successfully",
            "member_count": len(_MEMBER_CACHE)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/update_profile_photo", methods=["POST"])
def api_update_profile_photo():
    token = session.get('gym_token')
    if not token:
        return jsonify({"ok": False, "error": "Not logged in. Please login first."}), 401

    data = request.get_json(force=True)
    image_b64 = data.get("image_b64")
    if not image_b64:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    try:
        # Prepare the request payload - try different formats
        payload = {
            "api_key": GYM_API_KEY,
            "token": token,
            "memberphoto": f"data:image/jpeg;base64,{image_b64}"
        }
        
        print(f"DEBUG: Updating profile photo to GymMaster...")
        print(f"DEBUG: API Key: {GYM_API_KEY[:10]}...")
        print(f"DEBUG: Token: {token[:20]}...")
        print(f"DEBUG: Image size: {len(image_b64)} characters")
        
        # Based on API documentation: memberphoto accepts jpg, png, or base64 encoded string
        # Type: formData file or string
        success = False
        error_msg = ""
        data = {}
        
        # Try the working endpoint (endpoint 3) with correct format
        endpoint = f"{GYM_BASE_URL}/portal/api/v1/member/profile"
        
        # Approach 1: Form data with file upload (recommended by API docs)
        try:
            print("DEBUG: Trying Form data with file upload (recommended approach)")
            files = {
                'memberphoto': ('photo.jpg', base64.b64decode(image_b64), 'image/jpeg')
            }
            form_data = {
                'api_key': GYM_API_KEY,
                'token': token
            }
            r = requests.post(endpoint, data=form_data, files=files, timeout=30)
            r.raise_for_status()
            data = r.json()
            print(f"DEBUG: Form data response: {data}")
            if data.get("error") is None:
                success = True
        except Exception as e:
            error_msg = f"Form data failed: {str(e)}"
            print(f"DEBUG: {error_msg}")
        
        # Approach 2: Form data with base64 string
        if not success:
            try:
                print("DEBUG: Trying Form data with base64 string")
                form_data = {
                    'api_key': GYM_API_KEY,
                    'token': token,
                    'memberphoto': image_b64
                }
                r = requests.post(endpoint, data=form_data, timeout=30)
                r.raise_for_status()
                data = r.json()
                print(f"DEBUG: Form data base64 response: {data}")
                if data.get("error") is None:
                    success = True
            except Exception as e:
                error_msg = f"Form data base64 failed: {str(e)}"
                print(f"DEBUG: {error_msg}")
        
        # Approach 3: JSON with base64 (fallback)
        if not success:
            try:
                print("DEBUG: Trying JSON with base64 (fallback)")
                payload2 = {
                    "api_key": GYM_API_KEY,
                    "token": token,
                    "memberphoto": image_b64
                }
                r = requests.post(endpoint, json=payload2, timeout=30)
                r.raise_for_status()
                data = r.json()
                print(f"DEBUG: JSON base64 response: {data}")
                if data.get("error") is None:
                    success = True
            except Exception as e:
                error_msg = f"JSON base64 failed: {str(e)}"
                print(f"DEBUG: {error_msg}")
        
        if not success:
            return jsonify({
                "ok": False, 
                "error": f"All approaches failed. Last error: {error_msg}"
            })
        
        print(f"DEBUG: Final GymMaster response: {data}")
        
        if data.get("error") is None:
            # Clear cached profile data to force reload
            session.pop('profile_data', None)
            return jsonify({
                "ok": True, 
                "message": "Profile photo updated successfully",
                "response": data.get("result", {})
            })
        else:
            return jsonify({
                "ok": False, 
                "error": f"GymMaster API error: {data.get('error')}"
            })
            
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Request error: {e}")
        return jsonify({"ok": False, "error": f"Failed to connect to GymMaster: {str(e)}"})
    except Exception as e:
        print(f"DEBUG: Unexpected error: {e}")
        return jsonify({"ok": False, "error": f"Unexpected error: {str(e)}"})


@app.route("/api/register_face", methods=["POST"])
def api_register_face():
    token = session.get('gym_token')
    if not token:
        return jsonify({"ok": False, "error": "Not logged in. Please login first."}), 401

    # Get user email from session or profile
    profile = session.get('profile_data')
    if not profile:
        profile = gym_get_profile(token)
        if profile:
            session['profile_data'] = profile
    
    if not profile or not isinstance(profile, dict):
        return jsonify({"ok": False, "error": "Failed to get user profile"})
    
    user_email = profile.get("email")
    user_name = profile.get("fullname", "Unknown User")
    if not user_email:
        return jsonify({"ok": False, "error": "No email found in profile"})

    data = request.get_json(force=True)
    frames = data.get("frames", [])
    if not frames:
        return jsonify({"ok": False, "error": "No frames provided"})

    print(f"DEBUG: Registering face for {user_email} with {len(frames)} frames")

    # Process all frames to extract embeddings
    embeddings = []
    for i, frame_b64 in enumerate(frames):
        bgr = b64_to_bgr(frame_b64)
        if bgr is not None:
            emb = extract_embedding(bgr)
            if emb is not None:
                embeddings.append(emb)
                print(f"DEBUG: Frame {i+1}: Face detected and embedding extracted")
            else:
                print(f"DEBUG: Frame {i+1}: No face detected")
        else:
            print(f"DEBUG: Frame {i+1}: Invalid image")

    if not embeddings:
        return jsonify({"ok": False, "error": "No faces detected in any frame"})

    # Average all embeddings to create a single representative embedding
    avg_embedding = np.mean(embeddings, axis=0)
    # Normalize the averaged embedding
    norm = np.linalg.norm(avg_embedding) + 1e-8
    avg_embedding = avg_embedding / norm

    print(f"DEBUG: Created average embedding from {len(embeddings)} faces")

    # Save to database
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Convert embedding to JSON string
        embedding_json = json.dumps(avg_embedding.tolist())
        
        # Check if user already exists by email in profile data
        # We need to find the member by email from GymMaster profile
        cur.execute("SELECT id FROM member WHERE first_name = %s AND last_name = %s", 
                   (profile.get("firstname", ""), profile.get("surname", "")))
        existing = cur.fetchone()
        
        if existing:
            # Update existing record
            cur.execute(
                "UPDATE member SET enc = %s WHERE id = %s",
                (embedding_json, existing[0])
            )
            print(f"DEBUG: Updated existing record for {user_name} (ID: {existing[0]})")
        else:
            # Insert new record with member_id from profile
            cur.execute(
                "INSERT INTO member (member_id, first_name, last_name, enc) VALUES (%s, %s, %s, %s)",
                (profile.get("id", 0), profile.get("firstname", ""), profile.get("surname", ""), embedding_json)
            )
            print(f"DEBUG: Created new record for {user_name}")
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Clear cache to force reload
        invalidate_member_cache()
        
        return jsonify({
            "ok": True, 
            "message": f"Face recognition registered successfully for {user_email}",
            "frames_processed": len(frames),
            "faces_detected": len(embeddings)
        })
        
    except Exception as e:
        print(f"DEBUG: Database error: {e}")
        return jsonify({"ok": False, "error": f"Database error: {str(e)}"})


# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
