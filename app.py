from datetime import datetime
import base64
import hmac
import hashlib
import io
import json
import os
import pickle
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import cv2
import mysql.connector
import numpy as np
import redis
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
# ==================== STARTUP OPTIMIZATION ====================
def startup_optimization():
    """Preload components at startup for better performance"""
    print("DEBUG: Starting performance optimization...")
    
    # Preload InsightFace model
    try:
        print("1. Preloading InsightFace model...")
        load_insightface()
        print("   ✓ Model preloaded")
    except Exception as e:
        print(f"   ✗ Model preload failed: {e}")
    
    # Preload member cache
    try:
        print("2. Preloading member cache...")
        ensure_cache_loaded(force_refresh=True)
        print(f"   ✓ Cache preloaded with {len(_MEMBER_CACHE)} members")
    except Exception as e:
        print(f"   ✗ Cache preload failed: {e}")
    
    # Test Redis connection
    try:
        print("3. Testing Redis connection...")
        # DIOPTIMALKAN: Check if get_redis_conn is available
        if 'get_redis_conn' in globals():
            redis_conn = get_redis_conn()
            if redis_conn:
                print("   ✓ Redis connected")
            else:
                print("   ✗ Redis connection failed")
        else:
            print("   ⚠️ Redis function not available yet")
    except Exception as e:
        print(f"   ✗ Redis test failed: {e}")
    
    print("DEBUG: Startup optimization completed!")



# Call startup optimization when module loads
# MOVED TO END OF FILE - after all functions are defined

# DIOPTIMALKAN: Warm-up system untuk menghindari cold start
# from warmup_system import start_warmup_system, stop_warmup_system, get_warmup_status

# Auto-start warm-up system (DISABLED sementara untuk debugging)
# start_warmup_system()
# ================================================================


# -------------------- Config & Globals --------------------
load_dotenv()

APP_PORT = int(os.getenv("PORT", 8080))
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
# Ensure SECRET_KEY is bytes
if isinstance(SECRET_KEY, str):
    SECRET_KEY = SECRET_KEY.encode("utf-8")

# Security settings for door selection
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ALLOWED_IPS = os.getenv("ALLOWED_IPS", "127.0.0.1,192.168.1.0/24").split(",")
DOOR_ACCESS_SECRET = os.getenv("DOOR_ACCESS_SECRET", "DOOR_SECRET_2024")

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

# Rate Limiting & Security Configuration
RATE_LIMIT_STORAGE_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
LOGIN_LOCKOUT_TIME = int(os.getenv("LOGIN_LOCKOUT_TIME", "300"))  # 5 minutes
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "100 per minute")
LOGIN_RATE_LIMIT = os.getenv("LOGIN_RATE_LIMIT", "5 per minute")

# Multi-user per branch configuration
BRANCH_RATE_LIMIT = os.getenv("BRANCH_RATE_LIMIT", "50 per minute")  # Per branch
USER_RATE_LIMIT = os.getenv("USER_RATE_LIMIT", "10 per minute")  # Per user
MAX_USERS_PER_BRANCH = int(os.getenv("MAX_USERS_PER_BRANCH", "20"))  # Max concurrent users

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
    first_name: Optional[str]
    last_name: Optional[str]
    enc: np.ndarray  # 512-d embedding


# DIOPTIMALKAN: Thresholds for ArcFace cosine similarity - lebih permisif
SIM_THRESHOLD_MATCH = float(os.getenv("ARC_COS_THRESHOLD", "0.40"))
TOP2_MARGIN = float(os.getenv("TOP2_MARGIN", "0.06"))


def load_insightface():
    global _face_rec_model, _face_det
    try:
        with _insightface_lock:
            if _face_rec_model is None:
                print("DEBUG: Loading InsightFace model...")
                from insightface.app import FaceAnalysis
                _face_rec_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # Larger model with better accuracy
                _face_rec_model.prepare(ctx_id=0, det_size=(224, 224))  # DIOPTIMALKAN: Smaller detection size for speed
                _face_det = _face_rec_model  # detector is part of FaceAnalysis
                print("DEBUG: InsightFace model loaded successfully")
        return _face_rec_model, _face_det
    except Exception as e:
        print(f"DEBUG: Error loading InsightFace model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


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
            password=REDIS_PASSWORD if REDIS_PASSWORD else None,
            decode_responses=False,
            max_connections=10,
            socket_connect_timeout=2,  # Reduced timeout
            socket_timeout=2,  # Reduced timeout
            retry_on_timeout=False,  # Don't retry on timeout
            health_check_interval=0  # Disable health check
        )
        # Test connection with short timeout
        r.ping()
        return r
    except (redis.ConnectionError, redis.TimeoutError, ConnectionRefusedError) as e:
        # Redis connection failed - return None gracefully
        print(f"DEBUG: Redis connection failed: {e}")
        return None
    except Exception as e:
        # Other Redis errors
        print(f"DEBUG: Redis error: {e}")
        return None


# -------------------- Brute Force Protection Functions --------------------

def get_client_identifier():
    """Get unique client identifier for rate limiting"""
    # Try to get real IP from headers (for reverse proxy setups)
    client_ip = request.headers.get('X-Forwarded-For', request.headers.get('X-Real-IP', get_remote_address()))
    if ',' in client_ip:
        client_ip = client_ip.split(',')[0].strip()
    return client_ip

def is_login_blocked(identifier: str) -> bool:
    """Check if login is blocked for this identifier"""
    redis_conn = get_redis_conn()
    if not redis_conn:
        return False
    
    try:
        key = f"login_attempts:{identifier}"
        attempts = redis_conn.get(key)
        if attempts:
            return int(attempts) >= MAX_LOGIN_ATTEMPTS
        return False
    except Exception as e:
        print(f"DEBUG: Error checking login block status: {e}")
        return False

def increment_login_attempts(identifier: str) -> int:
    """Increment login attempts for this identifier"""
    redis_conn = get_redis_conn()
    if not redis_conn:
        return 0
    
    try:
        key = f"login_attempts:{identifier}"
        attempts = redis_conn.incr(key)
        if attempts == 1:  # First attempt, set expiration
            redis_conn.expire(key, LOGIN_LOCKOUT_TIME)
        return attempts
    except Exception as e:
        print(f"DEBUG: Error incrementing login attempts: {e}")
        return 0

def reset_login_attempts(identifier: str):
    """Reset login attempts for this identifier (on successful login)"""
    redis_conn = get_redis_conn()
    if not redis_conn:
        return
    
    try:
        key = f"login_attempts:{identifier}"
        redis_conn.delete(key)
    except Exception as e:
        print(f"DEBUG: Error resetting login attempts: {e}")

def get_remaining_lockout_time(identifier: str) -> int:
    """Get remaining lockout time in seconds"""
    redis_conn = get_redis_conn()
    if not redis_conn:
        return 0
    
    try:
        key = f"login_attempts:{identifier}"
        ttl = redis_conn.ttl(key)
        return max(0, ttl) if ttl > 0 else 0
    except Exception as e:
        print(f"DEBUG: Error getting lockout time: {e}")
        return 0

def get_user_identifier(email: str = None, user_agent: str = None) -> str:
    """Get unique user identifier for multi-user rate limiting"""
    client_ip = get_client_identifier()
    
    # If email is provided, use email + IP for user-specific tracking
    if email:
        return f"user:{email}:{client_ip}"
    
    # If user agent is provided, use it for browser fingerprinting
    if user_agent:
        import hashlib
        user_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        return f"browser:{user_hash}:{client_ip}"
    
    # Fallback to IP only
    return f"ip:{client_ip}"

def get_branch_identifier() -> str:
    """Get branch identifier (IP-based)"""
    client_ip = get_client_identifier()
    return f"branch:{client_ip}"

def check_branch_capacity(branch_id: str) -> bool:
    """Check if branch has capacity for more users"""
    redis_conn = get_redis_conn()
    if not redis_conn:
        return True  # Allow if Redis is not available
    
    try:
        key = f"branch_users:{branch_id}"
        current_users = redis_conn.scard(key)  # Get set cardinality
        return current_users < MAX_USERS_PER_BRANCH
    except Exception as e:
        print(f"DEBUG: Error checking branch capacity: {e}")
        return True

def add_user_to_branch(branch_id: str, user_id: str, ttl: int = 3600):
    """Add user to branch tracking (TTL in seconds)"""
    redis_conn = get_redis_conn()
    if not redis_conn:
        return
    
    try:
        key = f"branch_users:{branch_id}"
        redis_conn.sadd(key, user_id)
        redis_conn.expire(key, ttl)  # Auto-cleanup after 1 hour
    except Exception as e:
        print(f"DEBUG: Error adding user to branch: {e}")

def remove_user_from_branch(branch_id: str, user_id: str):
    """Remove user from branch tracking"""
    redis_conn = get_redis_conn()
    if not redis_conn:
        return
    
    try:
        key = f"branch_users:{branch_id}"
        redis_conn.srem(key, user_id)
    except Exception as e:
        print(f"DEBUG: Error removing user from branch: {e}")


def fetch_member_encodings() -> List[MemberEnc]:
    """
    Expected table structure (example):
      member (id BIGINT PK, gym_member_id BIGINT, email VARCHAR, enc MEDIUMBLOB)
    where `enc` stores a JSON array of float32 values (length 512).
    Standardized format: JSON only, no base64 NPY.
    """
    conn = None
    cur = None
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # First, check if enc column exists, if not create it
        try:
            # More reliable way to check column existence
            cur.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'member' 
                AND COLUMN_NAME = 'enc'
            """)
            column_exists = cur.fetchone()[0] > 0
            
            if not column_exists:
                print("DEBUG: Adding enc column to member table")
                # Use MEDIUMBLOB for better binary data handling (up to 16MB)
                # Standardized format: JSON array of float32 values
                cur.execute("ALTER TABLE member ADD COLUMN enc MEDIUMBLOB")
                conn.commit()
            else:
                print("DEBUG: enc column already exists")
        except Exception as e:
            if "Duplicate column name" in str(e):
                print("DEBUG: enc column already exists (duplicate error caught)")
            else:
                print(f"DEBUG: Error checking/creating enc column: {e}")
                # Continue anyway - column might exist
        
        cur.execute(
            """
            SELECT id AS member_pk,
            member_id AS gym_member_id,
            email,
            first_name,
            last_name,
            enc
            FROM member
            WHERE enc IS NOT NULL AND enc != ''
            """
        )
        out = []
        for member_pk, gym_id, email, first_name, last_name, enc_raw in cur.fetchall():
            if enc_raw is None:
                continue
            vec = None
            try:
                # Case A: BLOB (NPY bytes)
                if isinstance(enc_raw, (bytes, bytearray)):
                    vec = np.load(io.BytesIO(enc_raw), allow_pickle=False)
                else:
                    # Case B: JSON string array
                    if isinstance(enc_raw, str):
                        try:
                            vec = np.array(json.loads(enc_raw), dtype=np.float32)
                        except Exception:
                            # Case C: base64-encoded NPY string
                            vec = np.load(io.BytesIO(base64.b64decode(enc_raw)), allow_pickle=False)
                    else:
                        continue
                vec = vec.astype(np.float32).flatten()
                if vec.size == 0:
                    continue
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                out.append(MemberEnc(member_pk=member_pk, gym_member_id=gym_id or 0, email=email, first_name=first_name, last_name=last_name, enc=vec))
            except Exception as e:
                print(f"DEBUG: encoding load error for member {member_pk}: {e}")
                continue
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

# Matrix cache for fast cosine search (N,512)
_ENC_MATRIX: Optional[np.ndarray] = None

def _rebuild_matrix():
    """Rebuilds the in-memory (N,512) matrix for fast top-K scoring."""
    global _ENC_MATRIX
    if _MEMBER_CACHE:
        _ENC_MATRIX = np.stack([m.enc for m in _MEMBER_CACHE]).astype(np.float32)
    else:
        _ENC_MATRIX = None

# Throttling system to prevent spam
_LAST_RECOGNITION_TIME = {}  # {member_id: timestamp}
_RECOGNITION_COOLDOWN = 5  # 10 seconds cooldown per user

# Face recognition model cache
_face_rec_model = None
_face_det = None
_insightface_lock = threading.Lock()

# Cache refresh system - optimized for better performance
_LAST_CACHE_REFRESH = 0  # Timestamp of last cache refresh
_CACHE_REFRESH_INTERVAL = 3600  # 1 hour instead of 30 minutes  # 30 minutes - refresh cache every 30 minutes (increased from 5)


# Redis cache keys
REDIS_MEMBER_CACHE_KEY = "face_gate:member_encodings"
REDIS_PROFILE_CACHE_KEY = "face_gate:profile:{}"  # {} will be replaced with member_id
REDIS_CACHE_TTL = 7200  # 2 hours for better performance  # 1 hour cache TTL


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
    
    # If cache is already loaded and not force refresh, return immediately
    if _MEMBER_CACHE and not force_refresh:
        return
    
    current_time = time.time()
    
    # Check if we need to refresh cache (every 30 minutes)
    should_refresh = (
        force_refresh or 
        (current_time - _LAST_CACHE_REFRESH) > _CACHE_REFRESH_INTERVAL
    )
    
    # If force_refresh is True, skip cache and reload from database
    if should_refresh:
        print("DEBUG: Cache refresh needed, reloading from database...")
        _MEMBER_CACHE = []
        _ENC_MATRIX = None  # Clear matrix when force refreshing
        _LAST_CACHE_REFRESH = current_time
    
    if not _MEMBER_CACHE:
        # Try Redis first (only if not force refresh)
        if not should_refresh:
            cached_members = get_member_encodings_from_redis()
            if cached_members:
                _MEMBER_CACHE = cached_members
                print(f"DEBUG: Loaded {len(_MEMBER_CACHE)} members from Redis cache")
                _rebuild_matrix()  # Rebuild matrix after loading from Redis
                return  # Exit early if loaded from Redis
        
        # If no cache or force refresh, load from database
        if not _MEMBER_CACHE:
            try:
                print("DEBUG: Loading member encodings from database...")
                _MEMBER_CACHE = fetch_member_encodings()
                print(f"DEBUG: Loaded {len(_MEMBER_CACHE)} member encodings from database")
                
                # Save to Redis for next time
                if _MEMBER_CACHE:
                    save_member_encodings_to_redis(_MEMBER_CACHE)
                    _rebuild_matrix()  # Rebuild matrix after loading from database
                
                # Only print member details in debug mode
                if len(_MEMBER_CACHE) <= 5:  # Only print if small number of members
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
    """Return (best_member, best_score, second_best_score) using a single matmul."""
    try:
        ensure_cache_loaded()
        global _ENC_MATRIX
        if _ENC_MATRIX is None or _ENC_MATRIX.size == 0:
            return None, 0.0, 0.0
        scores = _ENC_MATRIX @ query_vec  # (N,)
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if scores.size > 1:
            # get second best without sorting full array
            second_best_score = float(np.partition(scores, -2)[-2])
        else:
            second_best_score = -1.0
        best_member = _MEMBER_CACHE[best_idx]
        return best_member, best_score, second_best_score
    except Exception as e:
        print(f"DEBUG: Error in find_best_match: {e}")
        return None, 0.0, 0.0


# -------------------- Security Functions --------------------

def is_ip_allowed(client_ip: str) -> bool:
    """Check if client IP is in allowed list"""
    import ipaddress
    try:
        client_ip_obj = ipaddress.ip_address(client_ip)
        for allowed_ip in ALLOWED_IPS:
            allowed_ip = allowed_ip.strip()
            if '/' in allowed_ip:
                # CIDR notation
                if client_ip_obj in ipaddress.ip_network(allowed_ip, strict=False):
                    return True
            else:
                # Single IP
                if str(client_ip_obj) == allowed_ip:
                    return True
        return False
    except Exception as e:
        print(f"DEBUG: IP check error: {e}")
        return False

def require_admin_auth(f):
    """Decorator to require admin authentication"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is admin authenticated
        if not session.get('admin_authenticated'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def require_door_access(f):
    """Decorator to require door access (IP + Secret)"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        secret_token = request.args.get("token")
        
        # Check IP
        if not is_ip_allowed(client_ip):
            print(f"DEBUG: Access denied for IP: {client_ip}")
            return "Access Denied - IP not allowed", 403
        
        # Check secret token
        if secret_token != DOOR_ACCESS_SECRET:
            print(f"DEBUG: Access denied - invalid token from IP: {client_ip}")
            return "Access Denied - Invalid token", 403
        
        print(f"DEBUG: Door access granted for IP: {client_ip}")
        return f(*args, **kwargs)
    return decorated_function

# -------------------- Door Token Functions --------------------

def _client_fingerprint(req) -> str:
    ua = req.headers.get("User-Agent", "")
    ip = req.headers.get("X-Forwarded-For", req.remote_addr or "")
    accept = req.headers.get("Accept", "")
    raw = f"{ua}|{ip}|{accept}"
    fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    print(f"DEBUG: Client fingerprint components - UA: {ua[:50]}..., IP: {ip}, Accept: {accept[:50]}...")
    print(f"DEBUG: Generated fingerprint: {fingerprint}")
    return fingerprint


def sign_door_token(doorid: int, device_fp: str, ttl_sec: int = 3600) -> str:  # 1 hour instead of 60 seconds
    exp = int(time.time()) + int(ttl_sec)
    payload = f"{doorid}.{device_fp}.{exp}"
    sig = hmac.new(SECRET_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_door_token(token: str, req) -> Optional[Tuple[int, str]]:
    try:
        print(f"DEBUG: Verifying token: {token[:30]}...")
        parts = token.split(".")
        if len(parts) != 4:
            print(f"DEBUG: Token has {len(parts)} parts, expected 4")
            return None
        doorid_s, device_fp, exp_s, sig = parts
        payload = f"{doorid_s}.{device_fp}.{exp_s}"
        exp = int(exp_s)
        current_time = int(time.time())
        print(f"DEBUG: Token exp: {exp}, current: {current_time}, expired: {current_time > exp}")
        
        sig_calc = hmac.new(SECRET_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        sig_match = hmac.compare_digest(sig, sig_calc)
        print(f"DEBUG: Signature match: {sig_match}")
        
        if not sig_match:
            return None
        if current_time > exp:
            print(f"DEBUG: Token expired")
            return None
        
        # bind to current client (temporarily disabled for testing)
        current_fp = _client_fingerprint(req)
        print(f"DEBUG: Token device_fp: {device_fp}, current_fp: {current_fp}")
        # Temporarily disable fingerprint check for testing
        # if device_fp != current_fp:
        #     print(f"DEBUG: Device fingerprint mismatch")
        #     return None
        return int(doorid_s), device_fp
    except Exception as e:
        print(f"DEBUG: Token verification error: {e}")
        return None


# -------------------- GymMaster API Helpers --------------------

def gym_login_with_memberid(memberid: int) -> Optional[str]:
    payload = {"api_key": GYM_API_KEY, "memberid": memberid}
    try:
        r = requests.post(GYM_LOGIN_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # Safe response handling
        if not data or data.get("error"):
            return None
        
        result = data.get("result") or {}
        token = result.get("token")
        return token
    except Exception:
        return None


def gym_login_with_email(email: str, password: str) -> Optional[Dict]:
    payload = {"api_key": GYM_API_KEY, "email": email, "password": password}
    try:
        r = requests.post(GYM_LOGIN_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # Safe response handling
        if not data or data.get("error"):
            return None
        
        result = data.get("result") or {}
        return result  # contains token, expires, memberid
    except Exception:
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
            
            # Check HTTP status before processing response
            status_code = r.status_code
            print(f"DEBUG: Endpoint {i+1} HTTP status: {status_code}")
            
            if status_code == 401:
                print(f"DEBUG: Endpoint {i+1} - 401 Unauthorized: Token invalid or expired")
                continue
            elif status_code == 403:
                print(f"DEBUG: Endpoint {i+1} - 403 Forbidden: Token valid but insufficient permissions")
                continue
            elif status_code == 404:
                print(f"DEBUG: Endpoint {i+1} - 404 Not Found: Endpoint doesn't exist (wrong URL)")
                continue
            elif status_code >= 400:
                print(f"DEBUG: Endpoint {i+1} - HTTP {status_code}: Other client error")
                continue
            
            r.raise_for_status()
            data = r.json()
            
            print(f"DEBUG: Gate API response: {data}")
            
            # Safe response handling
            if not data or data.get("error"):
                print(f"DEBUG: Gate API error: {data.get('error') if data else 'No response data'}")
            else:
                print(f"DEBUG: Gate opened successfully with endpoint {i+1}")
                result = data.get("result") or {}
                response = result.get("response")
                return response
                
        except requests.exceptions.HTTPError as e:
            print(f"DEBUG: Endpoint {i+1} HTTP error: {e}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Endpoint {i+1} request error: {e}")
            continue
        except Exception as e:
            print(f"DEBUG: Endpoint {i+1} unexpected error: {e}")
            continue
    
    print(f"DEBUG: All gate endpoints failed")
    print(f"DEBUG: Troubleshooting summary:")
    print(f"  - If you see 401 errors: Token is invalid/expired, check login")
    print(f"  - If you see 403 errors: Token valid but no gate permissions")
    print(f"  - If you see 404 errors: Wrong endpoint URLs, check GymMaster API docs")
    print(f"  - If you see timeouts: Network/connectivity issues")
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
            profile = json.loads(cached_data.decode("utf-8"))  # ← decode dulu
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
        r.setex(cache_key, REDIS_CACHE_TTL, json.dumps(profile).encode("utf-8"))
        print(f"DEBUG: Saved profile for member {member_id} to Redis cache")
    except Exception as e:
        print(f"DEBUG: Error saving profile to Redis: {e}")


def invalidate_member_cache():
    """Invalidate member encodings cache (both Redis and memory)"""
    global _MEMBER_CACHE, _ENC_MATRIX
    _MEMBER_CACHE = []
    _ENC_MATRIX = None  # Clear matrix when cache is invalidated
    
    r = get_redis_conn()
    if r:
        try:
            r.delete(REDIS_MEMBER_CACHE_KEY)
            print("DEBUG: Invalidated member encodings cache in Redis")
        except Exception as e:
            print(f"DEBUG: Error invalidating member cache in Redis: {e}")


def reload_member_cache():
    """Reload member cache and rebuild matrix after invalidation"""
    print("DEBUG: Reloading cache and rebuilding matrix after invalidation")
    ensure_cache_loaded(force_refresh=True)


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

# Enable CORS for API endpoints
CORS(app, origins=["*"])  # Configure this properly for production

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Dictionary to track connected devices
connected_devices = {}

# Initialize Rate Limiter with fallback
try:
    # Test Redis connection first
    test_redis = get_redis_conn()
    if test_redis:
        limiter = Limiter(
            app=app,
            key_func=get_client_identifier,
            storage_uri=RATE_LIMIT_STORAGE_URL,
            default_limits=[API_RATE_LIMIT],
            headers_enabled=True
        )
        print("DEBUG: Rate limiter initialized with Redis storage")
    else:
        # Fallback to in-memory storage
        limiter = Limiter(
            app=app,
            key_func=get_client_identifier,
            default_limits=[API_RATE_LIMIT],
            headers_enabled=True
        )
        print("DEBUG: Rate limiter initialized with in-memory storage (Redis unavailable)")
except Exception as e:
    # Fallback to basic rate limiter without storage
    limiter = Limiter(
        app=app,
        key_func=get_client_identifier,
        default_limits=[API_RATE_LIMIT],
        headers_enabled=True
    )
    print(f"DEBUG: Rate limiter initialized with fallback: {e}")

# Custom error handler for rate limiting
@limiter.request_filter
def custom_rate_limit_handler():
    """Custom handler for rate limit exceeded"""
    pass

@app.errorhandler(429)
def handle_rate_limit_exceeded(e):
    """Custom error handler for 429 Too Many Requests"""
    # Check if this is an API request
    if request.path.startswith('/api/'):
        return jsonify({
            "ok": False,
            "error": "Rate limit exceeded. Please try again later.",
            "retry_after": getattr(e, 'retry_after', 60)
        }), 429
    
    # For non-API requests, return HTML error page
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rate Limit Exceeded</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 50px; 
                background: #f8f9fa;
            }
            .error-container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 500px;
                margin: 0 auto;
            }
            .error-icon {
                font-size: 48px;
                color: #dc3545;
                margin-bottom: 20px;
            }
            .error-title {
                color: #dc3545;
                font-size: 24px;
                margin-bottom: 15px;
            }
            .error-message {
                color: #6c757d;
                margin-bottom: 20px;
            }
            .retry-info {
                background: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">⏱️</div>
            <div class="error-title">Rate Limit Exceeded</div>
            <div class="error-message">
                Too many requests. Please wait before trying again.
            </div>
            <div class="retry-info">
                <strong>Please wait 60 seconds</strong> before making another request.
            </div>
            <button onclick="window.location.reload()" style="
                background: #007bff; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer;
                margin-top: 20px;
            ">
                Try Again
            </button>
        </div>
    </body>
    </html>
    """), 429

@app.errorhandler(500)
def handle_internal_server_error(e):
    """Custom error handler for 500 Internal Server Error"""
    # Check if this is an API request
    if request.path.startswith('/api/'):
        return jsonify({
            "ok": False,
            "error": "Internal server error. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }), 500
    
    # For non-API requests, return HTML error page
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Internal Server Error</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 50px; 
                background: #f8f9fa;
            }
            .error-container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 500px;
                margin: 0 auto;
            }
            .error-icon {
                font-size: 48px;
                color: #dc3545;
                margin-bottom: 20px;
            }
            .error-title {
                color: #dc3545;
                font-size: 24px;
                margin-bottom: 15px;
            }
            .error-message {
                color: #6c757d;
                margin-bottom: 20px;
            }
            .retry-info {
                background: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">⚠️</div>
            <div class="error-title">Internal Server Error</div>
            <div class="error-message">
                Something went wrong on our end. Please try again later.
            </div>
            <div class="retry-info">
                <strong>Error Code:</strong> 500<br>
                <strong>Time:</strong> {{ timestamp }}
            </div>
            <button onclick="window.location.reload()" style="
                background: #007bff; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer;
                margin-top: 20px;
            ">
                Try Again
            </button>
        </div>
    </body>
    </html>
    """, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 500

# Ini untuk admin login saat mau ketik door id START
ADMIN_LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FTL Face Gate - Admin Login</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { 
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-card {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        .login-card h1 {
            color: #333;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
            box-sizing: border-box;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .login-btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .login-btn:hover {
            transform: translateY(-2px);
        }
        .error {
            color: #e74c3c;
            background: #fdf2f2;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #fecaca;
        }
        .security-note {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <h1><i class="fas fa-shield-alt"></i> Admin Access</h1>
        {% if error %}
        <div class="error">
            <i class="fas fa-exclamation-triangle"></i> {{ error }}
        </div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label for="username"><i class="fas fa-user"></i> Username</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password"><i class="fas fa-lock"></i> Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="login-btn">
                <i class="fas fa-sign-in-alt"></i> Login
            </button>
        </form>
        <div class="security-note">
            <i class="fas fa-info-circle"></i> 
            This is a secure admin area for door selection access.
        </div>
    </div>
</body>
</html>
"""
# Ini untuk admin login saat mau ketik door id END


# Ini untuk admin control panel untuk remote device management START
ADMIN_CONTROL_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FTL Face Gate - Device Control Panel</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            color: #333;
            font-size: 28px;
        }
        .header-actions {
            display: flex;
            gap: 10px;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-danger {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
        }
        .btn-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .stat-card .value {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }
        .devices-section {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .devices-section h2 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .device-list {
            display: grid;
            gap: 15px;
        }
        .device-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        .device-card:hover {
            border-color: #667eea;
            transform: translateX(5px);
        }
        .device-info {
            flex: 1;
        }
        .device-name {
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }
        .device-details {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            font-size: 14px;
            color: #666;
        }
        .device-detail {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .device-actions {
            display: flex;
            gap: 10px;
        }
        .btn-small {
            padding: 8px 16px;
            font-size: 12px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s;
            max-width: 400px;
        }
        @keyframes slideIn {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .notification.success { background: #10b981; }
        .notification.error { background: #ef4444; }
        .notification.info { background: #3b82f6; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state i {
            font-size: 64px;
            margin-bottom: 20px;
            opacity: 0.3;
        }
        .logout-btn {
            background: #6c757d;
            color: white;
        }
        .logout-btn:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-desktop"></i> Device Control Panel</h1>
            <div class="header-actions">
                <button class="btn btn-success" onclick="refreshAllDevices()">
                    <i class="fas fa-sync-alt"></i> Refresh All Devices
                </button>
                <button class="btn btn-primary" onclick="refreshDeviceList()">
                    <i class="fas fa-redo"></i> Reload List
                </button>
                <a href="/admin/logout" class="btn logout-btn">
                    <i class="fas fa-sign-out-alt"></i> Logout
                </a>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3><i class="fas fa-tablet-alt"></i> Connected Devices</h3>
                <div class="value" id="device-count">0</div>
            </div>
            <div class="stat-card">
                <h3><i class="fas fa-clock"></i> Last Update</h3>
                <div class="value" id="last-update" style="font-size: 18px;">Never</div>
            </div>
        </div>

        <div class="devices-section">
            <h2>
                <i class="fas fa-list"></i> Connected Devices
                <span class="status-indicator"></span>
            </h2>
            <div id="device-list" class="device-list">
                <div class="empty-state">
                    <i class="fas fa-tablet-alt"></i>
                    <p>Loading devices...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let notificationTimeout;

        function showNotification(message, type = 'info') {
            const existing = document.querySelector('.notification');
            if (existing) existing.remove();
            clearTimeout(notificationTimeout);

            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i> ${message}`;
            document.body.appendChild(notification);

            notificationTimeout = setTimeout(() => {
                notification.remove();
            }, 5000);
        }

        async function refreshDeviceList() {
            try {
                const response = await fetch('/api/admin/devices');
                const data = await response.json();
                
                if (data.ok) {
                    updateDeviceList(data.devices);
                    document.getElementById('device-count').textContent = data.total;
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                } else {
                    showNotification('Failed to load devices: ' + data.error, 'error');
                }
            } catch (error) {
                showNotification('Error loading devices: ' + error.message, 'error');
            }
        }

        function updateDeviceList(devices) {
            const listContainer = document.getElementById('device-list');
            
            if (devices.length === 0) {
                listContainer.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-tablet-alt"></i>
                        <p>No devices connected</p>
                    </div>
                `;
                return;
            }

            listContainer.innerHTML = devices.map(device => `
                <div class="device-card">
                    <div class="device-info">
                        <div class="device-name">
                            <i class="fas fa-tablet-alt"></i> ${device.name || 'Unknown Device'}
                        </div>
                        <div class="device-details">
                            <div class="device-detail">
                                <i class="fas fa-map-marker-alt"></i> ${device.location || 'Unknown'}
                            </div>
                            <div class="device-detail">
                                <i class="fas fa-door-open"></i> Door ID: ${device.doorid || 'N/A'}
                            </div>
                            <div class="device-detail">
                                <i class="fas fa-network-wired"></i> ${device.ip}
                            </div>
                            <div class="device-detail">
                                <i class="fas fa-clock"></i> ${new Date(device.connected_at).toLocaleString()}
                            </div>
                        </div>
                    </div>
                    <div class="device-actions">
                        <button class="btn btn-primary btn-small" onclick="refreshDevice('${device.id}')">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
            `).join('');
        }

        async function refreshAllDevices() {
            if (!confirm('Refresh all connected devices?')) return;
            
            try {
                const response = await fetch('/api/admin/broadcast_refresh', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: 'Admin requested refresh' })
                });
                const data = await response.json();
                
                if (data.ok) {
                    showNotification(data.message, 'success');
                } else {
                    showNotification('Failed to refresh devices: ' + data.error, 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }

        async function refreshDevice(clientId) {
            try {
                const response = await fetch('/api/admin/refresh_device', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ client_id: clientId })
                });
                const data = await response.json();
                
                if (data.ok) {
                    showNotification(data.message, 'success');
                } else {
                    showNotification('Failed to refresh device: ' + data.error, 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }

        // Initial load
        refreshDeviceList();

        // Auto-refresh every 5 seconds
        setInterval(refreshDeviceList, 5000);
    </script>
</body>
</html>
"""
# Ini untuk admin control panel untuk remote device management END




# Ini untuk main page Face Recognition START
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FTL Face Gate</title>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
  <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
  <style>
    body { 
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; 
      margin: 0; 
      padding: 20px;
      background: #f5f5f5;
      min-height: auto;
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
    
    /* Fix mirror effect for video */
    video {
      transform: scaleX(-1); /* Flip video horizontally to fix mirror effect */
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
    .card-stepper { 
      border: 1px solid #4ca7e5; 
      padding: 24px; 
      border-radius: 12px; 
      background: linear-gradient(135deg, #f8fbff 0%, #e8f4fd 100%); 
      box-shadow: 0 4px 12px rgba(76, 167, 229, 0.15); 
      width: 100%; 
      margin-bottom: 24px;
    }
    .muted { color: #777; }
    .field { display: inline-flex; gap: 8px; align-items: center; }
    .stack { display: grid; gap: 12px; }
    
    /* Profile Photo Display - Inside Camera Container */
    .profile-photo-container {
      position: absolute;
      bottom: 15px;
      right: 15px;
      width: 80px;
      height: 80px;
      border-radius: 50%;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
      background: linear-gradient(135deg, #4ca7e5 0%, #0072bc 100%);
      display: none;
      z-index: 1000;
      transition: all 0.3s ease;
      border: 3px solid rgba(255, 255, 255, 0.9);
    }
    
    .profile-photo-container.show {
      display: block;
      animation: slideInUp 0.5s ease-out;
    }
    
    .profile-photo-container img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 50%;
    }
    
    .profile-photo-placeholder {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: white;
      text-align: center;
    }
    
    .profile-photo-placeholder i {
      font-size: 2rem;
      margin-bottom: 5px;
      opacity: 0.8;
    }
    
    .profile-photo-placeholder span {
      font-size: 0.7rem;
      font-weight: 500;
    }
    
    .profile-name-overlay {
      position: absolute;
      bottom: -30px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 4px 8px;
      border-radius: 12px;
      font-size: 0.7rem;
      font-weight: 500;
      white-space: nowrap;
      max-width: 150px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    
    @keyframes slideInUp {
      from {
        transform: translateY(100px);
        opacity: 0;
      }
      to {
        transform: translateY(0);
        opacity: 1;
      }
    }
    
    /* Loading Animation for Recognizing State */
    @keyframes loadingPulse {
      0% {
        transform: scale(1);
        opacity: 0.8;
      }
      50% {
        transform: scale(1.1);
        opacity: 1;
      }
      100% {
        transform: scale(1);
        opacity: 0.8;
      }
    }
    
    @keyframes loadingRotate {
      0% {
        transform: rotate(0deg);
      }
      100% {
        transform: rotate(360deg);
      }
    }
    
    @keyframes loadingDots {
      0%, 20% {
        opacity: 0.5;
      }
      50% {
        opacity: 1;
      }
      80%, 100% {
        opacity: 0.5;
      }
    }
    
    @keyframes loadingScan {
      0% {
        transform: translateX(-100%);
        opacity: 0;
      }
      50% {
        opacity: 1;
      }
      100% {
        transform: translateX(100%);
        opacity: 0;
      }
    }
    
    .loading-animation {
      animation: loadingPulse 1.5s ease-in-out infinite;
    }
    
    .loading-icon {
      animation: loadingRotate 2s linear infinite;
    }
    
    .loading-text {
      animation: loadingDots 1.5s ease-in-out infinite;
    }
    
    .loading-scan-line {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 3px;
      background: linear-gradient(90deg, transparent, #2196F3, transparent);
      animation: loadingScan 2s ease-in-out infinite;
      z-index: 15;
    }
    
    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(33, 150, 243, 0.1);
      display: none;
      z-index: 10;
      pointer-events: none;
    }
    
    .loading-overlay.show {
      display: block;
    }
    
    .loading-content {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
      color: #2196F3;
      z-index: 20;
      padding: 20px;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
      min-width: 250px;
    }
    
    .loading-spinner {
      width: 40px;
      height: 40px;
      border: 4px solid rgba(33, 150, 243, 0.3);
      border-top: 4px solid #2196F3;
      border-radius: 50%;
      animation: loadingRotate 1s linear infinite;
      margin: 0 auto 20px;
    }
    
    .loading-text-large {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 15px;
      line-height: 1.2;
      text-shadow: none;
      color: #2196F3;
    }
    
    .loading-text-small {
      font-size: 14px;
      opacity: 0.8;
      line-height: 1.4;
      margin: 0;
    }
    
    /* Fullscreen loading animation */
    .fullscreen .loading-overlay {
      z-index: 10005 !important;
    }
    
    .fullscreen .loading-content {
      z-index: 10006 !important;
    }
    
    .fullscreen .loading-scan-line {
      z-index: 10007 !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
      body { margin: 10px; padding: 10px; }
      .row { flex-direction: column; gap: 16px; }
      #cameraContainer { 
        height: 400px !important; 
        min-height: 400px !important;
      }
      
      /* Profile photo responsive */
      .profile-photo-container {
        width: 60px;
        height: 60px;
        bottom: 10px;
        right: 10px;
      }
      
      .profile-photo-placeholder i {
        font-size: 1.2rem;
      }
      
      .profile-photo-placeholder span {
        font-size: 0.5rem;
      }
      
      .profile-name-overlay {
        font-size: 0.5rem;
        bottom: -20px;
        max-width: 100px;
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
        height: 500px !important; 
        min-height: 500px !important;
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 auto 16px auto !important;
      }
      video, canvas, img { 
        width: 100%; 
        height: 100%;
        object-fit: cover !important;
      }
      
      /* Profile photo responsive for small mobile */
      .profile-photo-container {
        width: 50px;
        height: 50px;
        bottom: 8px;
        right: 8px;
      }
      
      .profile-photo-placeholder i {
        font-size: 1rem;
      }
      
      .profile-photo-placeholder span {
        font-size: 0.4rem;
      }
      
      .profile-name-overlay {
        font-size: 0.4rem;
        bottom: -18px;
        max-width: 80px;
      }
      button { padding: 8px 12px; font-size: 12px; }
    }
    
    /* Mobile portrait optimization */
    @media (max-width: 414px) {
      #cameraContainer {
        height: 450px !important;
        min-height: 450px !important;
        width: 95% !important;
        margin: 0 auto 12px auto !important;
      }
    }
    
    /* Small mobile devices */
    @media (max-width: 375px) {
      #cameraContainer {
        height: 400px !important;
        min-height: 400px !important;
        width: 95% !important;
        margin: 0 auto 10px auto !important;
      }
    }
    
    /* Very small mobile devices */
    @media (max-width: 320px) {
      #cameraContainer {
        height: 350px !important;
        min-height: 350px !important;
        width: 95% !important;
        margin: 0 auto 8px auto !important;
      }
    }
    
    /* Mobile responsive for horizontal orientation */
    @media (max-width: 768px) {
      .video-horizontal {
        width: 100% !important;
        max-width: 100% !important;
        height: 300px !important;
        margin: 0 auto 16px auto !important;
      }
    }
    
    @media (max-width: 480px) {
      .video-horizontal {
        height: 250px !important;
        margin: 0 auto 12px auto !important;
      }
    }
    
    @media (max-width: 414px) {
      .video-horizontal {
        height: 220px !important;
        margin: 0 auto 10px auto !important;
      }
    }
    
    @media (max-width: 375px) {
      .video-horizontal {
        height: 200px !important;
        margin: 0 auto 8px auto !important;
      }
    }
    
    @media (max-width: 320px) {
      .video-horizontal {
        height: 180px !important;
        margin: 0 auto 6px auto !important;
      }
    }
    
    /* Video Orientation Styles */
    .video-vertical {
      width: 100% !important;
      max-width: 400px !important;
      height: 600px !important;
      margin: 0 auto 24px auto !important;
    }
    
    .video-horizontal {
      width: 100% !important;
      max-width: 600px !important;
      height: 400px !important;
      margin: 0 auto 24px auto !important;
    }
    
    /* Orientation Toggle Button Styles */
    #orientationToggle {
      transition: all 0.3s ease;
    }
    
    #orientationToggle:hover {
      background: #5a6268 !important;
      transform: translateY(-1px);
    }
    
    #orientationToggle.vertical {
      background: #6c757d !important;
    }
    
    #orientationToggle.horizontal {
      background: #007bff !important;
    }
    
    /* Fullscreen Styles */
    .fullscreen {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      z-index: 9999 !important;
      background: #000 !important;
    }
    
    .fullscreen #video {
      width: 100vw !important;
      height: 100vh !important;
      object-fit: contain !important;
      object-position: center !important;
      transform: scaleX(-1) !important;
      background: #000 !important;
    }
    
    /* Fullscreen responsive to video orientation */
    .fullscreen.video-vertical #video {
      object-fit: contain !important;
      object-position: center !important;
      width: 100vw !important;
      height: 100vh !important;
    }
    
    .fullscreen.video-horizontal #video {
      object-fit: contain !important;
      object-position: center !important;
      width: 100vw !important;
      height: 100vh !important;
    }
    
    /* Mobile fullscreen optimization */
    @media (max-width: 768px) {
      .fullscreen #video {
        object-fit: contain !important;
        object-position: center !important;
        width: 100vw !important;
        height: 100vh !important;
      }
    }
    
    .fullscreen .profile-photo-container {
      width: 120px;
      height: 120px;
      bottom: 30px;
      right: 30px;
      border: 4px solid rgba(255, 255, 255, 0.9);
    }
    
    .fullscreen .profile-photo-placeholder i {
      font-size: 3rem;
    }
    
    .fullscreen .profile-photo-placeholder span {
      font-size: 0.9rem;
    }
    
    .fullscreen .profile-name-overlay {
      font-size: 0.8rem;
      bottom: -35px;
      max-width: 180px;
    }
    
    /* Footer Styles */
    footer {
      text-align: center;
      padding: 20px;
      margin-top: 30px;
      background: #f8f9fa;
      border-top: 1px solid #e9ecef;
      color: #6c757d;
      font-size: 14px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial;
    }
    
    footer p {
      margin: 0;
      font-weight: 500;
    }
    
    /* Responsive Footer */
    @media (max-width: 768px) {
      footer {
        padding: 15px;
        margin-top: 20px;
        font-size: 13px;
      }
    }
    
    @media (max-width: 480px) {
      footer {
        padding: 12px;
        margin-top: 15px;
        font-size: 12px;
      }
    }
    
    .fullscreen {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 70vh !important;
      z-index: 9999 !important;
      background: black !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      margin: 0 !important;
    }
    
    /* Ensure fullscreen button is positioned absolutely and not affected by flexbox */
    .fullscreen #fullscreenBtn {
      position: fixed !important;
      top: 20px !important;
      right: 20px !important;
      z-index: 10001 !important;
      background: #495057 !important;
      color: white !important;
      border: none !important;
      padding: 12px 16px !important;
      border-radius: 8px !important;
      cursor: pointer !important;
      font-size: 16px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      min-width: 48px !important;
      height: 48px !important;
    }
    
    /* Hide browser default fullscreen exit button */
    .fullscreen::backdrop {
      display: none !important;
    }
    
    /* Hide any browser fullscreen UI elements */
    .fullscreen *::-webkit-media-controls-fullscreen-button,
    .fullscreen *::-webkit-media-controls-overlay-play-button,
    .fullscreen *::-webkit-media-controls-panel {
      display: none !important;
    }
    
    .fullscreen #video {
      width: 100vw !important;
      height: 100vh !important;
      max-width: none !important;
      min-height: 100vh !important;
      border-radius: 0 !important;
      object-fit: cover !important;
      transform: scaleX(-1) !important; /* Fix mirror effect in fullscreen */
    }
    
    /* Mobile fullscreen video - responsive to orientation */
    @media (max-width: 768px) {
      .fullscreen #video {
        width: 100vw !important;
        height: 100vh !important;
        object-fit: contain !important;
        max-height: 100vh !important;
        object-position: center !important;
        transform: scaleX(-1) !important; /* Fix mirror effect in mobile fullscreen */
        background: #000 !important;
      }
      
      /* If video is taller than screen, center it vertically */
      .fullscreen #video {
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) scaleX(-1) !important; /* Fix mirror effect in mobile fullscreen */
      }
      
      /* Mobile fullscreen overlay positioning - match 16:9 video */
      .fullscreen #overlay {
        width: 100vw !important;
        height: 56.25vw !important; /* Match video 16:9 aspect ratio */
        max-height: 100vh !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 15 !important;
      }
    }
    
    /* Tablet fullscreen video - responsive to orientation */
    @media (min-width: 769px) and (max-width: 1024px) {
      .fullscreen #video {
        width: 100vw !important;
        height: 100vh !important;
        object-fit: contain !important;
        max-height: 100vh !important;
        object-position: center !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) scaleX(-1) !important; /* Fix mirror effect in tablet fullscreen */
        background: #000 !important;
      }
      
      /* Tablet fullscreen overlay positioning - match 16:9 video */
      .fullscreen #overlay {
        width: 100vw !important;
        height: 56.25vw !important; /* Match video 16:9 aspect ratio */
        max-height: 100vh !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 15 !important;
      }
    }
    
    .fullscreen #overlay {
      width: 100vw !important;
      height: 56.25vw !important; /* 16:9 aspect ratio */
      max-height: 100vh !important;
      border-radius: 0 !important;
      position: absolute !important;
      top: 50% !important;
      left: 50% !important;
      transform: translate(-50%, -50%) !important;
    }
    
    .fullscreen #cameraStatus {
      top: 20px !important;
      left: 20px !important;
      font-size: 16px !important;
      padding: 10px 15px !important;
    }
    
    .fullscreen #fullscreenBtn {
      position: absolute !important;
      top: 20px !important;
      right: 20px !important;
      font-size: 16px !important;
      padding: 12px 16px !important;
      z-index: 10001 !important;
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
    
    /* Hide browser default fullscreen exit elements */
    .fullscreen .fullscreen-exit-browser,
    .fullscreen [data-fullscreen-exit],
    .fullscreen .browser-fullscreen-exit,
    .fullscreen [class*="fullscreen-exit"]:not(#fullscreenExitBtn),
    .fullscreen [id*="fullscreen-exit"]:not(#fullscreenExitBtn) {
      display: none !important;
    }
    
    /* Hide browser fullscreen UI overlay */
    .fullscreen::before,
    .fullscreen::after {
      display: none !important;
    }
    
    /* Hide any browser-generated fullscreen UI */
    .fullscreen *[class*="fullscreen-exit"],
    .fullscreen *[id*="fullscreen-exit"]:not(#fullscreenExitBtn),
    .fullscreen [data-fullscreen],
    .fullscreen [data-exit-fullscreen] {
      display: none !important;
    }
    
    /* Ensure our custom exit button is always visible */
    #fullscreenExitBtn {
      display: flex !important;
    }
    
    /* Fullscreen popup styles - let popup_style determine colors */
    .fullscreen #facePopup {
      position: fixed !important;
      top: 20% !important;
      left: 50% !important;
      transform: translate(-50%, -50%) !important;
      z-index: 10002 !important;
      /* background color will be set by popup_style */
      color: white !important;
      padding: 20px 30px !important;
      border-radius: 12px !important;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
      text-align: center !important;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
      min-width: 300px !important;
      max-width: 400px !important;
      animation: popupSlideIn 0.3s ease-out !important;
    }
    
    /* Fullscreen countdown timer styles */
    .fullscreen #centerCountdown {
      position: absolute !important;
      top: 50% !important;
      left: 50% !important;
      transform: translate(-50%, -50%) !important;
      z-index: 10002 !important;
      background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(20,20,20,0.95)) !important;
      backdrop-filter: blur(10px) !important;
      border: 2px solid rgba(255,255,255,0.1) !important;
      border-radius: 25px !important;
      padding: 40px 50px !important;
      box-shadow: 0 20px 60px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.05) !important;
      text-align: center !important;
      color: white !important;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
      min-width: 200px !important;
      animation: countdownPulse 0.5s ease-out !important;
    }
    
    /* Enhanced fullscreen countdown with special class */
    .fullscreen #centerCountdown.fullscreen-countdown {
      background: linear-gradient(135deg, rgba(0,0,0,0.95), rgba(30,30,30,0.98)) !important;
      border: 3px solid rgba(255,255,255,0.2) !important;
      box-shadow: 0 25px 80px rgba(0,0,0,0.9), 0 0 0 2px rgba(255,255,255,0.1), inset 0 1px 0 rgba(255,255,255,0.1) !important;
      animation: countdownPulse 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    
    .fullscreen #centerCountdownNumber {
      font-size: 72px !important;
      font-weight: 900 !important;
      margin-bottom: 15px !important;
      background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4) !important;
      background-size: 400% 400% !important;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
      background-clip: text !important;
      animation: gradientShift 2s ease-in-out infinite, numberPulse 1s ease-in-out infinite !important;
      text-shadow: 0 0 30px rgba(255,255,255,0.3) !important;
    }
    
    .fullscreen #centerCountdown div:last-child {
      font-size: 20px !important;
      opacity: 0.9 !important;
      font-weight: 500 !important;
      letter-spacing: 0.5px !important;
      text-transform: uppercase !important;
    }
    
    /* Fullscreen countdown message styling */
    .fullscreen #countdownMessage {
      font-size: 16px !important;
      opacity: 0.8 !important;
      font-weight: 400 !important;
      letter-spacing: 0.3px !important;
      margin-top: 12px !important;
      color: rgba(255,255,255,0.9) !important;
    }
    
    /* Enhanced fullscreen countdown message */
    .fullscreen #centerCountdown.fullscreen-countdown #countdownMessage {
      font-size: 18px !important;
      opacity: 0.9 !important;
      font-weight: 500 !important;
      color: rgba(255,255,255,0.95) !important;
      text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }
    
    /* Fullscreen countdown animations */
    @keyframes countdownPulse {
      0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
      50% { transform: translate(-50%, -50%) scale(1.05); opacity: 0.8; }
      100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    }
    
    @keyframes gradientShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    
    @keyframes numberPulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.1); }
      100% { transform: scale(1); }
    }
    
    /* Mobile fullscreen countdown adjustments */
    @media (max-width: 768px) {
      .fullscreen #centerCountdown {
        padding: 30px 40px !important;
        border-radius: 20px !important;
        min-width: 180px !important;
      }
      
      .fullscreen #centerCountdownNumber {
        font-size: 60px !important;
        margin-bottom: 12px !important;
      }
      
      .fullscreen #centerCountdown div:last-child {
        font-size: 18px !important;
      }
      
      .fullscreen #countdownMessage {
        font-size: 14px !important;
        margin-top: 10px !important;
      }
      
      .fullscreen #centerCountdown.fullscreen-countdown #countdownMessage {
        font-size: 16px !important;
      }
    }
    
    /* Tablet fullscreen countdown adjustments */
    @media (min-width: 769px) and (max-width: 1024px) {
      .fullscreen #centerCountdown {
        padding: 35px 45px !important;
        border-radius: 22px !important;
      }
      
      .fullscreen #centerCountdownNumber {
        font-size: 66px !important;
      }
      
      .fullscreen #countdownMessage {
        font-size: 17px !important;
        margin-top: 11px !important;
      }
      
      .fullscreen #centerCountdown.fullscreen-countdown #countdownMessage {
        font-size: 19px !important;
      }
    }
      pointer-events: none !important;
    }
    
    /* Fullscreen Exit Button */
    .fullscreen-exit {
      position: absolute !important;
      top: 20px !important;
      right: 20px !important;
      background: rgba(220, 53, 69, 0.9) !important;
      color: white !important;
      border: none !important;
      padding: 12px 20px !important;
      border-radius: 8px !important;
      cursor: pointer !important;
      font-size: 16px !important;
      font-weight: 600 !important;
      z-index: 10000 !important;
      min-width: 140px !important;
      height: 48px !important;
      text-align: center !important;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      gap: 8px !important;
      transition: all 0.2s ease !important;
    }
    
    .fullscreen-exit:hover {
      background: rgba(220, 53, 69, 1) !important;
    }
    
    /* Mobile-specific fullscreen exit button */
    @media (max-width: 768px) {
      .fullscreen-exit {
        top: 15px !important;
        right: 15px !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        min-width: 120px !important;
        height: 44px !important;
        border-radius: 6px !important;
      }
      
      .fullscreen #fullscreenBtn {
        position: fixed !important;
        top: 15px !important;
        right: 15px !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        height: 44px !important;
        border-radius: 6px !important;
        z-index: 10001 !important;
        background: #495057 !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 44px !important;
      }
    }
    
    /* High resolution mobile screens (1080x2436, iPhone X style) */
    @media (max-width: 414px) and (min-height: 800px) {
      .fullscreen-exit {
        top: 20px !important;
        right: 20px !important;
        padding: 12px 18px !important;
        font-size: 16px !important;
        min-width: 130px !important;
        height: 48px !important;
        z-index: 10001 !important;
        border-radius: 8px !important;
      }
      
      .fullscreen #fullscreenBtn {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        font-size: 16px !important;
        padding: 12px 18px !important;
        height: 48px !important;
        border-radius: 8px !important;
        z-index: 10001 !important;
        background: #495057 !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 48px !important;
      }
      
      .fullscreen #cameraStatus {
        top: 20px !important;
        left: 20px !important;
        font-size: 18px !important;
        padding: 12px 18px !important;
        z-index: 10001 !important;
      }
    }
    
    /* Ultra-wide mobile screens */
    @media (min-width: 400px) and (max-width: 500px) and (min-height: 800px) {
      .fullscreen-exit {
        top: 25px !important;
        right: 25px !important;
        padding: 14px 20px !important;
        font-size: 18px !important;
        min-width: 140px !important;
        height: 52px !important;
        border-radius: 8px !important;
      }
      
      .fullscreen #fullscreenBtn {
        position: fixed !important;
        top: 25px !important;
        right: 25px !important;
        font-size: 18px !important;
        padding: 14px 20px !important;
        height: 52px !important;
        border-radius: 8px !important;
        z-index: 10001 !important;
        background: #495057 !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 52px !important;
      }
    }
  </style>
</head>
<body>
    <div style="max-width: 800px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); padding: 32px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="margin: 0 0 8px 0; font-size: 28px; font-weight: 700; color: #333;">FTL Face Recognition Gate</h1>
        <div id="doorInfo" style="display: flex; align-items: center; gap: 8px; justify-content: center; display: none;">
            <span style="color: #666; font-size: 14px;">Door ID:</span>
          <span id="doorid" class="pill" style="background: #495057; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;">{{ doorid or '—' }}</span>
        </div>
      </div>
      
        <!-- Camera Display Area -->
        <div id="cameraContainer" class="video-vertical" style="position: relative; width: 100%; max-width: 400px; height: 600px; background: #f8f9fa; border-radius: 12px; margin: 0 auto 24px auto; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <video id="video" autoplay playsinline muted style="width: 100%; height: 100%; background: #f8f9fa; border-radius: 12px; object-fit: cover; border: none; transform: scaleX(-1); display: none;"></video>
            <canvas id="overlay" style="position: absolute; top: 0; left: 0; pointer-events: none; border-radius: 12px; width: 100%; height: 100%; z-index: 10; background: transparent;"></canvas>
            
            <!-- Loading Overlay for Recognizing State -->
            <div id="loadingOverlay" class="loading-overlay">
              <div class="loading-scan-line"></div>
              <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text-large">Scanning...</div>
                <div class="loading-text-small">Please wait while we scan your face</div>
              </div>
            </div>
            
            <!-- Center Countdown Animation - MOVED INSIDE CAMERA CONTAINER FOR FULLSCREEN -->
            <div id="centerCountdown" style="display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 10002; text-align: center; background: rgba(0,0,0,0.8); color: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
              <div style="font-size: 48px; font-weight: bold; margin-bottom: 10px;" id="centerCountdownNumber">3</div>
              
              <div style="font-size: 14px; opacity: 0.6; margin-top: 8px;" id="countdownMessage">Tunggu sebentar untuk scan berikutnya</div>
            </div>
          
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
          
          <!-- Profile Photo Container - Inside Camera -->
          <div id="profilePhotoContainer" class="profile-photo-container">
            <div id="profilePhotoPlaceholder" class="profile-photo-placeholder">
              <i class="fas fa-user"></i>
              <span>Profile</span>
            </div>
            <img id="profilePhoto" style="display: none; transform: scaleX(-1);" alt="Profile Photo">
            <div id="profileNameOverlay" class="profile-name-overlay">            </div>
          </div>
        </div>
        
        <!-- Orientation Toggle Button -->
        <div style="text-align: center; margin-bottom: 16px;">
          <button id="orientationToggle" onclick="toggleVideoOrientation()" style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 20px; font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 6px; margin: 0 auto;">
            <i class="fas fa-mobile-alt"></i>
            <span>Vertical</span>
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
      </div>
      
      <!-- Status Cards - Simplified -->
      <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px;">
        <!-- Face Detection Status -->
        <div id="detectionResult" style="padding: 12px; border-radius: 8px; background: #f8f9fa; border: 1px solid #e9ecef; display: flex; align-items: center; gap: 8px;">
          <div style="width: 24px; height: 24px; border-radius: 50%; background: #dc3545; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">
            <i class="fas fa-times"></i>
          </div>
          <div id="detectedName" style="font-size: 14px; font-weight: 500; color: #333;">No face detected</div>
        </div>
        
        <!-- Camera Status -->
        <div id="cameraStatusCard" style="padding: 12px; border-radius: 8px; background: #f8f9fa; border: 1px solid #e9ecef; display: flex; align-items: center; gap: 8px;">
          <div id="cameraStatusIcon" style="width: 24px; height: 24px; border-radius: 50%; background: #dc3545; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">
            <i class="fas fa-times"></i>
          </div>
          <div id="detectionStatus" style="font-size: 14px; font-weight: 500; color: #333;">Camera inactive</div>
        </div>
        
        <!-- Countdown Timer -->
        <div id="countdownTimer" style="display: none; padding: 12px; border-radius: 8px; background: #fff3cd; border: 1px solid #ffeaa7; display: flex; align-items: center; gap: 8px; transition: all 0.3s ease; opacity: 0; transform: translateY(10px);">
          <div style="width: 24px; height: 24px; border-radius: 50%; background: #ffc107; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">
            <i class="fas fa-clock"></i>
          </div>
          <div style="font-size: 14px; font-weight: 500; color: #856404;">Next scan: <span id="countdownSeconds" style="font-weight: bold; color: #d63384;">0</span>s</div>
        </div>
      </div>
      
      <div style="color: #999; font-size: 14px; text-align: center; margin-top: 16px;">
        Face recognition runs automatically when camera is active.
      </div>
    </div>
  <script>
    // Get doorid from backend template (hidden from URL)
    const doorid = '{{ doorid }}' || null;
    const DOOR_TOKEN = '{{ door_token or "" }}' || null;
    
    // Debug logging
    console.log('DEBUG: doorid =', doorid);
    console.log('DEBUG: DOOR_TOKEN =', DOOR_TOKEN ? DOOR_TOKEN.substring(0, 20) + '...' : 'null');
    
    // Function to refresh door token when expired
    async function refreshDoorToken() {
      if (!doorid) return null;
      try {
        const response = await fetch('/api/access_gate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ doorid: parseInt(doorid) })
        });
        const result = await response.json();
        if (result.ok) {
          // Reload page to get new token
          window.location.reload();
        }
      } catch (error) {
        console.error('Failed to refresh token:', error);
      }
    }
    
    // Hide door info from UI
    const doorInfo = document.getElementById('doorInfo');
    if (doorInfo) {
      doorInfo.style.display = 'none';
    }
    
    // Update doorid display if needed (but keep hidden)
    const dooridElement = document.getElementById('doorid');
    if (dooridElement && doorid) {
      dooridElement.textContent = doorid;
    }

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
      let hasSuccessfulScan = false; // Flag to prevent multiple successful scans
      let noFaceCount = 0; // Counter for consecutive no face detections
      let isPopupShowing = false; // Flag to prevent multiple popups
      let lastNoFaceResetTime = 0; // Track when noFaceCount was last reset
      let lastImageData = null; // Track last image data to detect changes

    function setLog(obj) {
      console.log(typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2));
    }
    
    // Simple image similarity calculation
    function calculateImageSimilarity(img1, img2) {
      try {
        // Simple comparison based on data URL length and hash
        if (img1 === img2) return 1.0;
        
        // Extract base64 data
        const data1 = img1.split(',')[1];
        const data2 = img2.split(',')[1];
        
        if (!data1 || !data2) return 0.0;
        
        // Simple length-based similarity
        const len1 = data1.length;
        const len2 = data2.length;
        const lengthDiff = Math.abs(len1 - len2) / Math.max(len1, len2);
        
        // If length difference is too large, images are very different
        if (lengthDiff > 0.3) return 0.0;
        
        // Simple character-based similarity
        let matches = 0;
        const minLen = Math.min(len1, len2);
        for (let i = 0; i < minLen; i++) {
          if (data1[i] === data2[i]) matches++;
        }
        
        return matches / minLen;
      } catch (error) {
        console.error('Error calculating image similarity:', error);
        return 0.0;
      }
    }
    
    // Progress Roadmap Functions
    function updateProgressRoadmap() {
      let completedSteps = 0;
      
      // Check if user is logged in (Step 1: Validasi)
      const isLoggedIn = sessionStorage.getItem('gym_token') || localStorage.getItem('gym_token');
      if (isLoggedIn) {
        completedSteps = 1;
        updateStepStatus(1, true);
      } else {
        updateStepStatus(1, false);
      }
      
      // Check if face recognition is registered (Step 2: Face Recognition)
      // This would be checked via API call to see if user has face data
      // For now, we'll assume it's completed if user is logged in and has done face recognition
      if (completedSteps >= 1) {
        // Simulate face recognition completion - in real app, check via API
        const hasFaceData = sessionStorage.getItem('face_registered') || false;
        if (hasFaceData) {
          completedSteps = 2;
          updateStepStatus(2, true);
        } else {
          updateStepStatus(2, false);
        }
      } else {
        updateStepStatus(2, false);
      }
      
      // Check if photo is uploaded to GymMaster (Step 3: Upload Foto)
      if (completedSteps >= 2) {
        // Simulate photo upload completion - in real app, check via API
        const hasPhotoUploaded = sessionStorage.getItem('photo_uploaded') || false;
        if (hasPhotoUploaded) {
          completedSteps = 3;
          updateStepStatus(3, true);
        } else {
          updateStepStatus(3, false);
        }
      } else {
        updateStepStatus(3, false);
      }
      
      // Update progress line
      updateProgressLine(completedSteps);
    }
    
    
    function updateStepStatus(stepNumber, isCompleted) {
      const stepElement = document.querySelector(`[data-step="${stepNumber}"]`);
      if (!stepElement) {
        console.warn(`Step element not found for step ${stepNumber}`);
        return;
      }
      
      const circle = stepElement.querySelector('.step-circle');
      if (!circle) {
        console.warn(`Circle element not found for step ${stepNumber}`);
        return;
      }
      
      const checkIcon = circle.querySelector('.fas.fa-check');
      const stepNumberSpan = circle.querySelector('.step-number');
      
      if (isCompleted) {
        // FTL GYM hero colors for completed steps
        circle.style.background = 'linear-gradient(135deg, #4ca7e5 0%, #0072bc 50%, #0037cf 100%)';
        circle.style.borderColor = '#0037cf';
        circle.style.boxShadow = '0 6px 16px rgba(0, 55, 207, 0.4)';
        circle.style.color = 'white';
        
        if (checkIcon) checkIcon.style.display = 'block';
        if (stepNumberSpan) stepNumberSpan.style.display = 'none';
        
        // Update text color to match FTL GYM colors
        const stepText = stepElement.querySelector('div:last-child');
        if (stepText) {
          stepText.style.color = '#0037cf';
          stepText.style.fontWeight = '600';
        }
      } else {
        circle.style.background = '#f8f9fa';
        circle.style.borderColor = '#e9ecef';
        circle.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
        circle.style.color = '#6c757d';
        
        if (checkIcon) checkIcon.style.display = 'none';
        if (stepNumberSpan) stepNumberSpan.style.display = 'block';
        circle.style.animation = 'none';
        
        // Reset text color
        const stepText = stepElement.querySelector('div:last-child');
        if (stepText) {
          stepText.style.color = '#6c757d';
          stepText.style.fontWeight = '500';
        }
      }
    }
    
    function updateProgressLine(completedSteps) {
      const progressLine = document.getElementById('progressLine');
      if (!progressLine) {
        console.warn('Progress line element not found');
        return;
      }
      
      const totalSteps = 3;
      const progressPercentage = (completedSteps / totalSteps) * 100;
      
      progressLine.style.width = progressPercentage + '%';
      progressLine.style.background = 'linear-gradient(90deg, #4ca7e5 0%, #0072bc 50%, #0037cf 100%)';
      progressLine.style.boxShadow = '0 4px 12px rgba(0, 55, 207, 0.4)';
    }
    
    // Bounding box functionality removed

    function clearFaceIndicator() {
      if (!overlay) return;
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function syncOverlayWithVideo() {
      if (!overlay || !video) return;
      
      const displayWidth = video.clientWidth;
      const displayHeight = video.clientHeight;
      
      // Check if we're in fullscreen mode
      const isFullscreen = document.getElementById('cameraContainer').classList.contains('fullscreen');
      
      if (isFullscreen) {
        // In fullscreen, use full screen dimensions
        const maxWidth = window.innerWidth;
        const maxHeight = window.innerHeight;
        
        let overlayWidth = maxWidth;
        let overlayHeight = maxHeight;
        
        overlay.width = overlayWidth;
        overlay.height = overlayHeight;
        overlay.style.width = overlayWidth + 'px';
        overlay.style.height = overlayHeight + 'px';
        
        console.log('Fullscreen overlay synced with 16:9 ratio:', overlayWidth, 'x', overlayHeight);
      } else {
        // Normal mode - match video display size exactly
        overlay.width = displayWidth;
        overlay.height = displayHeight;
        overlay.style.width = displayWidth + 'px';
        overlay.style.height = displayHeight + 'px';
        
        console.log('Overlay synced with video:', displayWidth, 'x', displayHeight);
      }
    }
    
    // Enhanced sync function for mobile
    function forceSyncOverlay() {
      if (!overlay || !video) return;
      
      // Wait for video to be ready
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        setTimeout(forceSyncOverlay, 100);
        return;
      }
      
      const displayWidth = video.clientWidth;
      const displayHeight = video.clientHeight;
      
      // Check if we're in fullscreen mode
      const isFullscreen = document.getElementById('cameraContainer').classList.contains('fullscreen');
      
      if (isFullscreen) {
        // In fullscreen, use full screen dimensions
        const maxWidth = window.innerWidth;
        const maxHeight = window.innerHeight;
        
        let overlayWidth = maxWidth;
        let overlayHeight = maxHeight;
        
        overlay.width = overlayWidth;
        overlay.height = overlayHeight;
        overlay.style.width = overlayWidth + 'px';
        overlay.style.height = overlayHeight + 'px';
        
        console.log('Force sync completed with 16:9 ratio:', overlayWidth, 'x', overlayHeight);
      } else {
        // Normal mode - force overlay dimensions
        overlay.width = displayWidth;
        overlay.height = displayHeight;
        overlay.style.width = displayWidth + 'px';
        overlay.style.height = displayHeight + 'px';
        
        console.log('Force sync completed:', displayWidth, 'x', displayHeight);
      }
      
      // Clear and redraw
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function setButtons(running) {
      btnStart.disabled = running || !DOOR_TOKEN;
      btnStop.disabled = !running;
    }

    function showPopup(name, status, popupStyle) {
      // Don't show popup if one is already showing
      if (isPopupShowing) {
        console.log('Skipping popup - already showing:', name, status, popupStyle);
        return;
      }
      
      // Remove existing popup if any
      const existingPopup = document.getElementById('facePopup');
      if (existingPopup) {
        existingPopup.remove();
      }
      
      // Don't show popup if we're in no-face mode (noFaceCount >= 3)
      if (noFaceCount >= 3) {
        console.log('Skipping popup due to no-face mode:', name, status, popupStyle);
        return;
      }
      
      // Set flag to prevent multiple popups
      isPopupShowing = true;
      
      // Create popup element
      const popup = document.createElement('div');
      popup.id = 'facePopup';
      
      // Set colors based on popup_style
      let backgroundColor, textColor, icon;
      switch(popupStyle.toUpperCase()) {
        case 'GRANTED':
          backgroundColor = '#4CAF50'; // Green
          textColor = '#FFFFFF';
          icon = 'fas fa-check-circle';
          break;
        case 'DENIED':
          backgroundColor = '#F44336'; // Red
          textColor = '#FFFFFF';
          icon = 'fas fa-times-circle';
          break;
        case 'WARNING':
          backgroundColor = '#FF9800'; // Orange
          textColor = '#FFFFFF';
          icon = 'fas fa-exclamation-triangle';
          break;
        default:
          backgroundColor = '#2196F3'; // Blue (default)
          textColor = '#FFFFFF';
          icon = 'fas fa-info-circle';
      }
      
      popup.style.cssText = `
        position: fixed;
        top: 20%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: ${backgroundColor};
        color: ${textColor};
        padding: 20px 30px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        z-index: 10002;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-width: 300px;
        max-width: 400px;
        animation: popupSlideIn 0.3s ease-out;
        pointer-events: none;
      `;
      
      popup.innerHTML = `
        <div style="font-size: 48px; margin-bottom: 15px;">
          <i class="${icon}"></i>
        </div>
        <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">
          ${name || 'Unknown'}
        </div>
        <div style="font-size: 16px; opacity: 0.9;">
          ${status}
        </div>
      `;
      
      // Add responsive styles for mobile
      const mobileStyles = document.createElement('style');
      mobileStyles.id = 'mobilePopupStyles';
      mobileStyles.textContent = `
        @media (max-width: 768px) {
          #facePopup {
            min-width: 280px !important;
            max-width: 90vw !important;
            padding: 16px 24px !important;
            top: 15% !important;
          }
          #facePopup div:first-child {
            font-size: 40px !important;
            margin-bottom: 12px !important;
          }
          #facePopup div:nth-child(2) {
            font-size: 20px !important;
            margin-bottom: 8px !important;
          }
          #facePopup div:last-child {
            font-size: 14px !important;
          }
        }
        @media (max-width: 414px) {
          #facePopup {
            min-width: 260px !important;
            padding: 14px 20px !important;
            top: 10% !important;
          }
          #facePopup div:first-child {
            font-size: 36px !important;
          }
          #facePopup div:nth-child(2) {
            font-size: 18px !important;
          }
          #facePopup div:last-child {
            font-size: 13px !important;
          }
        }
        
        /* Fullscreen popup responsive styles */
        .fullscreen #facePopup {
          z-index: 10002 !important;
        }
        
        @media (max-width: 768px) {
          .fullscreen #facePopup {
            min-width: 280px !important;
            max-width: 90vw !important;
            padding: 16px 24px !important;
            top: 15% !important;
          }
          .fullscreen #facePopup div:first-child {
            font-size: 40px !important;
            margin-bottom: 12px !important;
          }
          .fullscreen #facePopup div:nth-child(2) {
            font-size: 20px !important;
            margin-bottom: 8px !important;
          }
          .fullscreen #facePopup div:last-child {
            font-size: 14px !important;
          }
        }
        
        @media (max-width: 414px) {
          .fullscreen #facePopup {
            min-width: 260px !important;
            padding: 14px 20px !important;
            top: 10% !important;
          }
          .fullscreen #facePopup div:first-child {
            font-size: 36px !important;
          }
          .fullscreen #facePopup div:nth-child(2) {
            font-size: 18px !important;
          }
          .fullscreen #facePopup div:last-child {
            font-size: 13px !important;
          }
        }
      `;
      if (!document.getElementById('mobilePopupStyles')) {
        document.head.appendChild(mobileStyles);
      }
      
      // Add CSS animation
      if (!document.getElementById('popupStyles')) {
        const style = document.createElement('style');
        style.id = 'popupStyles';
        style.textContent = `
          @keyframes popupSlideIn {
            from {
              opacity: 0;
              transform: translate(-50%, -50%) scale(0.8);
            }
            to {
              opacity: 1;
              transform: translate(-50%, -50%) scale(1);
            }
          }
        `;
        document.head.appendChild(style);
      }
      
      // Check if we're in fullscreen mode and append to the right container
      const cameraContainer = document.getElementById('cameraContainer');
      const isFullscreen = cameraContainer && cameraContainer.classList.contains('fullscreen');
      
      if (isFullscreen) {
        // In fullscreen mode, append to cameraContainer to ensure it's visible
        cameraContainer.appendChild(popup);
        console.log('Popup added to fullscreen container');
      } else {
        // Normal mode, append to body
        document.body.appendChild(popup);
        console.log('Popup added to body');
      }
      
      // Auto remove after 3 seconds to prevent multiple popups
      setTimeout(() => {
        if (popup && popup.parentNode) {
          popup.style.animation = 'popupSlideIn 0.3s ease-out reverse';
          setTimeout(() => {
            if (popup && popup.parentNode) {
              popup.remove();
              isPopupShowing = false; // Reset flag when popup is removed
              console.log('Popup auto-removed after 3 seconds');
            }
          }, 300);
        }
      }, 3000);
    }

    function updateDetectionDisplay(name, status, confidence = null, popupStyle = null) {
      const detectionResult = document.getElementById('detectionResult');
      const statusIcon = detectionResult.querySelector('i');
      const statusCircle = detectionResult.querySelector('div');
      
      // Update camera status card
      const cameraStatusCard = document.getElementById('cameraStatusCard');
      const cameraStatusIcon = document.getElementById('cameraStatusIcon');
      const cameraStatusIconElement = cameraStatusIcon.querySelector('i');
      
      detectedName.textContent = name || 'No face detected';
      
      // Show popup based on popup_style
      if (popupStyle) {
        showPopup(name, status, popupStyle);
      }
      detectionStatus.textContent = status;
      
      // Check if status is "Scanning..." to show loading animation
      const isRecognizing = status === 'Scanning...';
      const loadingOverlay = document.getElementById('loadingOverlay');
      
      // Show/hide loading overlay - only if not in no-face mode
      if (loadingOverlay) {
        if (isRecognizing && noFaceCount < 3) {
          loadingOverlay.classList.add('show');
        } else {
          loadingOverlay.classList.remove('show');
        }
      }
      
      // Don't update status display if we're in no-face mode and status is "Scanning..."
      if (isRecognizing && noFaceCount >= 3) {
        console.log('Skipping "Scanning..." status update due to no-face mode (count:', noFaceCount, ')');
        return;
      }
      
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
      const centerCountdown = document.getElementById('centerCountdown');
      const centerCountdownNumber = document.getElementById('centerCountdownNumber');
      const countdownMessage = document.getElementById('countdownMessage');
      
      console.log('showCenterCountdown called with:', seconds);
      
      // Ensure seconds is a positive integer
      const validSeconds = Math.max(1, Math.floor(seconds));
      console.log('Validated seconds:', validSeconds);
      
      if (validSeconds > 0) {
        // Clear any existing interval
        if (countdownInterval) {
          clearInterval(countdownInterval);
        }
        
        // Check if we're in fullscreen mode for enhanced animations
        const isFullscreen = document.getElementById('cameraContainer').classList.contains('fullscreen');
        console.log('Countdown in fullscreen mode:', isFullscreen);
        
        // Show center countdown animation
        centerCountdown.style.display = 'block';
        centerCountdown.style.opacity = '0';
        centerCountdown.style.transform = 'translate(-50%, -50%) scale(0.8)';
        
        // Enhanced animation for fullscreen mode
        if (isFullscreen) {
          // Add special fullscreen countdown class for enhanced styling
          centerCountdown.classList.add('fullscreen-countdown');
          
          // Update message for fullscreen mode
          if (countdownMessage) {
            countdownMessage.textContent = 'Tunggu untuk scan berikutnya';
          }
          
          // Animate in with enhanced effect
          setTimeout(() => {
            centerCountdown.style.transition = 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
            centerCountdown.style.opacity = '1';
            centerCountdown.style.transform = 'translate(-50%, -50%) scale(1)';
          }, 10);
        } else {
          // Standard animation for normal mode
          if (countdownMessage) {
            countdownMessage.textContent = 'Tunggu sebentar untuk scan berikutnya';
          }
          
          setTimeout(() => {
            centerCountdown.style.transition = 'all 0.3s ease';
            centerCountdown.style.opacity = '1';
            centerCountdown.style.transform = 'translate(-50%, -50%) scale(1)';
          }, 10);
        }
        
        centerCountdownNumber.textContent = validSeconds;
        remainingCooldown = validSeconds;
        
        // Start countdown with enhanced animation
        countdownInterval = setInterval(() => {
          remainingCooldown--;
          
          // Ensure countdown never goes below 0
          if (remainingCooldown < 0) {
            remainingCooldown = 0;
          }
          
          centerCountdownNumber.textContent = remainingCooldown;
          
          // Enhanced pulse animation for fullscreen mode
          if (isFullscreen) {
            // More dramatic animation for fullscreen
            centerCountdownNumber.style.transform = 'scale(1.3) rotate(5deg)';
            centerCountdownNumber.style.transition = 'all 0.2s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
            
            setTimeout(() => {
              centerCountdownNumber.style.transform = 'scale(1) rotate(0deg)';
              centerCountdownNumber.style.transition = 'all 0.3s ease';
            }, 200);
            
            // Add special effects for last 3 seconds
            if (remainingCooldown <= 3) {
              centerCountdownNumber.style.animation = 'numberPulse 0.5s ease-in-out infinite';
              centerCountdown.style.boxShadow = '0 0 50px rgba(255, 107, 107, 0.5), 0 20px 60px rgba(0,0,0,0.8)';
              
              // Update message for final countdown
              if (countdownMessage) {
                countdownMessage.textContent = 'Hampir selesai! Scan akan segera tersedia';
                countdownMessage.style.color = 'rgba(255, 107, 107, 0.9)';
              }
            } else if (remainingCooldown <= 5) {
              // Update message for 5 seconds remaining
              if (countdownMessage) {
                countdownMessage.textContent = 'Tinggal beberapa detik lagi...';
                countdownMessage.style.color = 'rgba(255, 255, 255, 0.9)';
              }
            }
          } else {
            // Standard animation for normal mode
            centerCountdownNumber.style.transform = 'scale(1.2)';
            setTimeout(() => {
              centerCountdownNumber.style.transform = 'scale(1)';
            }, 150);
          }
          
          console.log('Center countdown:', remainingCooldown);
          
          if (remainingCooldown <= 0) {
            hideCountdownTimer();
          }
        }, 1000);
      } else {
        hideCountdownTimer();
      }
    }

      function hideCountdownTimer() {
        console.log('hideCountdownTimer called');
        
        // Hide center countdown animation
        const centerCountdown = document.getElementById('centerCountdown');
        if (centerCountdown) {
          // Check if we're in fullscreen mode for enhanced exit animation
          const isFullscreen = document.getElementById('cameraContainer').classList.contains('fullscreen');
          
          if (isFullscreen) {
            // Enhanced exit animation for fullscreen
            centerCountdown.style.transition = 'all 0.5s cubic-bezier(0.55, 0.055, 0.675, 0.19)';
            centerCountdown.style.opacity = '0';
            centerCountdown.style.transform = 'translate(-50%, -50%) scale(0.5) rotate(10deg)';
            
            // Remove fullscreen countdown class
            centerCountdown.classList.remove('fullscreen-countdown');
          } else {
            // Standard exit animation for normal mode
            centerCountdown.style.transition = 'all 0.3s ease';
            centerCountdown.style.opacity = '0';
            centerCountdown.style.transform = 'translate(-50%, -50%) scale(0.8)';
          }
          
          setTimeout(() => {
            centerCountdown.style.display = 'none';
            // Reset any special effects
            centerCountdown.style.boxShadow = '';
            const centerCountdownNumber = document.getElementById('centerCountdownNumber');
            const countdownMessage = document.getElementById('countdownMessage');
            if (centerCountdownNumber) {
              centerCountdownNumber.style.animation = '';
              centerCountdownNumber.style.transform = '';
            }
            if (countdownMessage) {
              countdownMessage.style.color = '';
              countdownMessage.textContent = 'Tunggu sebentar untuk scan berikutnya';
            }
          }, isFullscreen ? 500 : 300);
        }
        
        if (countdownInterval) {
          clearInterval(countdownInterval);
          countdownInterval = null;
        }
        remainingCooldown = 0;
        
            // Reset successful scan flag and restart recognition interval after cooldown ends
            hasSuccessfulScan = false; // Reset flag to allow scanning again
            noFaceCount = 0; // Reset no face counter
            console.log('Cooldown ended, resetting hasSuccessfulScan flag and noFaceCount');
            
            // Show completion message briefly
            const countdownMessage = document.getElementById('countdownMessage');
            if (countdownMessage) {
              countdownMessage.textContent = 'Cooldown selesai! Siap untuk scan berikutnya';
              countdownMessage.style.color = 'rgba(76, 175, 80, 0.9)';
            }
        
        if (stream && DOOR_TOKEN && !recognitionInterval) {
          recognitionInterval = setInterval(performRecognition, 500);
          console.log('Recognition interval restarted after cooldown');
          
          // Reset countdown message after a brief delay
          setTimeout(() => {
            const countdownMessage = document.getElementById('countdownMessage');
            if (countdownMessage) {
              countdownMessage.textContent = 'Tunggu sebentar untuk scan berikutnya';
              countdownMessage.style.color = '';
            }
          }, 2000);
        } else {
          console.log('Cannot restart recognition - stream:', !!stream, 'token:', !!DOOR_TOKEN, 'interval:', !!recognitionInterval);
        }
      }

      async function toggleFullscreen() {
        const cameraContainer = document.getElementById('cameraContainer');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        
        try {
          if (!document.fullscreenElement) {
            // Enter fullscreen using browser API (like F11)
            if (cameraContainer.requestFullscreen) {
              await cameraContainer.requestFullscreen();
            } else if (cameraContainer.webkitRequestFullscreen) {
              await cameraContainer.webkitRequestFullscreen();
            } else if (cameraContainer.mozRequestFullScreen) {
              await cameraContainer.mozRequestFullScreen();
            } else if (cameraContainer.msRequestFullscreen) {
              await cameraContainer.msRequestFullscreen();
            }
            
            // Add CSS classes for styling
            cameraContainer.classList.add('fullscreen');
            fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
            
            // Maintain video orientation in fullscreen
            const isVertical = cameraContainer.classList.contains('video-vertical');
            if (isVertical) {
              cameraContainer.classList.add('video-vertical');
            } else {
              cameraContainer.classList.add('video-horizontal');
            }
            
            // Add exit button
            
            
            // Hide any browser-generated fullscreen UI elements
            const cleanupBrowserUI = () => {
              // Remove any browser-generated fullscreen exit buttons
              const browserExitElements = document.querySelectorAll('[class*="fullscreen-exit"]:not(#fullscreenExitBtn), [id*="fullscreen-exit"]:not(#fullscreenExitBtn), [data-fullscreen], [data-exit-fullscreen]');
              browserExitElements.forEach(el => {
                if (el.id !== 'fullscreenExitBtn') {
                  el.style.display = 'none';
                  el.remove();
                }
              });
              
              // Remove any browser fullscreen overlays
              const overlays = document.querySelectorAll('[class*="fullscreen-overlay"], [class*="browser-fullscreen"]');
              overlays.forEach(el => el.remove());
            };
            
            // Run cleanup immediately and periodically
            cleanupBrowserUI();
            setTimeout(cleanupBrowserUI, 100);
            setTimeout(cleanupBrowserUI, 500);
            
            // Set up periodic cleanup while in fullscreen
            const cleanupInterval = setInterval(cleanupBrowserUI, 1000);
            
            // Store interval ID for cleanup
            window.fullscreenCleanupInterval = cleanupInterval;
            
            isFullscreen = true;
            console.log('Entered browser fullscreen mode (like F11)');
          } else {
            // Exit fullscreen
            if (document.exitFullscreen) {
              await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
              await document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
              await document.mozCancelFullScreen();
            } else if (document.msExitFullscreen) {
              await document.msExitFullscreen();
            }
            
            // Remove CSS classes but maintain orientation
            cameraContainer.classList.remove('fullscreen');
            fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
            
            // Ensure orientation classes are maintained
            const isVertical = cameraContainer.classList.contains('video-vertical');
            if (!isVertical && !cameraContainer.classList.contains('video-horizontal')) {
              // Default to vertical if no orientation class
              cameraContainer.classList.add('video-vertical');
            }
            
            // Remove exit button
            const exitBtn = document.getElementById('fullscreenExitBtn');
            if (exitBtn) {
              exitBtn.remove();
            }
            
            // Clear cleanup interval
            if (window.fullscreenCleanupInterval) {
              clearInterval(window.fullscreenCleanupInterval);
              window.fullscreenCleanupInterval = null;
            }
            
            isFullscreen = false;
            console.log('Exited browser fullscreen mode');
          }
          
          // Sync overlay after fullscreen change
          setTimeout(() => {
            if (overlay && video) {
              syncOverlayWithVideo();
            }
          }, 100);
          
        } catch (error) {
          console.error('Fullscreen error:', error);
          // Fallback to CSS fullscreen if browser API fails
          if (!isFullscreen) {
            cameraContainer.classList.add('fullscreen');
            isFullscreen = true;
            console.log('Fallback to CSS fullscreen');
          } else {
            cameraContainer.classList.remove('fullscreen');
            isFullscreen = false;
            console.log('Fallback exit CSS fullscreen');
          }
        }
      }

      function handleFullscreenChange() {
        // Handle browser fullscreen API changes
        const isCurrentlyFullscreen = !!(document.fullscreenElement || 
                                        document.webkitFullscreenElement || 
                                        document.mozFullScreenElement || 
                                        document.msFullscreenElement);
        
        if (!isCurrentlyFullscreen && isFullscreen) {
          // User exited fullscreen via browser controls (ESC key, etc.)
          const cameraContainer = document.getElementById('cameraContainer');
          const fullscreenBtn = document.getElementById('fullscreenBtn');
          
          cameraContainer.classList.remove('fullscreen');
          fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i> Fullscreen';
          
          // Ensure orientation classes are maintained
          const isVertical = cameraContainer.classList.contains('video-vertical');
          if (!isVertical && !cameraContainer.classList.contains('video-horizontal')) {
            // Default to vertical if no orientation class
            cameraContainer.classList.add('video-vertical');
          }
          
          // Remove exit button
          const exitBtn = document.getElementById('fullscreenExitBtn');
          if (exitBtn) {
            exitBtn.remove();
          }
          
          // Clear cleanup interval
          if (window.fullscreenCleanupInterval) {
            clearInterval(window.fullscreenCleanupInterval);
            window.fullscreenCleanupInterval = null;
          }
          
          isFullscreen = false;
          console.log('Exited fullscreen via browser controls (ESC, etc.)');
          
          // Sync overlay
          setTimeout(() => {
            if (overlay && video) {
              syncOverlayWithVideo();
            }
          }, 100);
        }
      }

    // Video orientation toggle function
    function toggleVideoOrientation() {
      const cameraContainer = document.getElementById('cameraContainer');
      const toggleButton = document.getElementById('orientationToggle');
      const icon = toggleButton.querySelector('i');
      const text = toggleButton.querySelector('span');
      const video = document.getElementById('video');
      const deviceInfo = getDeviceOrientation();
      
      // Check if camera is currently active
      const isCameraActive = video && video.srcObject && !video.paused;
      
      // Set manual override flag for tablets
      if (deviceInfo.isTablet) {
        cameraContainer.dataset.manualOverride = 'true';
        console.log('Tablet manual override enabled');
      }
      
      if (cameraContainer.classList.contains('video-vertical')) {
        // Switch to horizontal
        cameraContainer.classList.remove('video-vertical');
        cameraContainer.classList.add('video-horizontal');
        toggleButton.classList.remove('vertical');
        toggleButton.classList.add('horizontal');
        icon.className = 'fas fa-desktop';
        text.textContent = deviceInfo.isTablet ? 'Manual Horizontal' : 'Horizontal';
        console.log('Switched to horizontal orientation');
      } else {
        // Switch to vertical
        cameraContainer.classList.remove('video-horizontal');
        cameraContainer.classList.add('video-vertical');
        toggleButton.classList.remove('horizontal');
        toggleButton.classList.add('vertical');
        icon.className = 'fas fa-mobile-alt';
        text.textContent = deviceInfo.isTablet ? 'Manual Vertical' : 'Vertical';
        console.log('Switched to vertical orientation');
      }
      
      // Update camera constraints based on orientation
      updateCameraConstraints();
      
      // If camera is active, restart it with new constraints
      if (isCameraActive) {
        console.log('Restarting camera with new orientation...');
        
        // Show loading indicator
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
          loadingOverlay.style.display = 'flex';
        }
        
        stopCam().then(() => {
          setTimeout(() => {
            startCam().then(() => {
              // Hide loading indicator after camera starts
              if (loadingOverlay) {
                loadingOverlay.style.display = 'none';
              }
            }).catch((error) => {
              console.error('Failed to restart camera:', error);
              if (loadingOverlay) {
                loadingOverlay.style.display = 'none';
              }
            });
          }, 500); // Small delay to ensure clean restart
        });
      }
    }
    
    // Detect device orientation
    function getDeviceOrientation() {
      const isPortrait = window.innerHeight > window.innerWidth;
      const isTablet = window.innerWidth >= 768 && window.innerWidth <= 1024;
      const isMobile = window.innerWidth < 768;
      
      console.log('Device info:', {
        width: window.innerWidth,
        height: window.innerHeight,
        isPortrait: isPortrait,
        isTablet: isTablet,
        isMobile: isMobile
      });
      
      return { isPortrait, isTablet, isMobile };
    }
    
    // Update orientation button text based on device and current state
    function updateOrientationButton() {
      const toggleButton = document.getElementById('orientationToggle');
      if (!toggleButton) return;
      
      const deviceInfo = getDeviceOrientation();
      const icon = toggleButton.querySelector('i');
      const text = toggleButton.querySelector('span');
      const cameraContainer = document.getElementById('cameraContainer');
      const hasManualOverride = cameraContainer.dataset.manualOverride === 'true';
      
      if (deviceInfo.isTablet) {
        if (hasManualOverride) {
          // Show manual override state
          const isVerticalMode = cameraContainer.classList.contains('video-vertical');
          if (isVerticalMode) {
            icon.className = 'fas fa-mobile-alt';
            text.textContent = 'Manual Vertical';
            toggleButton.title = 'Manual vertical mode - click to switch to horizontal';
          } else {
            icon.className = 'fas fa-desktop';
            text.textContent = 'Manual Horizontal';
            toggleButton.title = 'Manual horizontal mode - click to switch to vertical';
          }
        } else {
          // Show auto-detection state
          if (deviceInfo.isPortrait) {
            icon.className = 'fas fa-mobile-alt';
            text.textContent = 'Auto Vertical';
            toggleButton.title = 'Auto-detected portrait mode - click to override';
          } else {
            icon.className = 'fas fa-desktop';
            text.textContent = 'Auto Horizontal';
            toggleButton.title = 'Auto-detected landscape mode - click to override';
          }
        }
      } else {
        // For mobile and desktop, use button state
        const isVerticalMode = cameraContainer.classList.contains('video-vertical');
        if (isVerticalMode) {
          icon.className = 'fas fa-mobile-alt';
          text.textContent = 'Vertical';
          toggleButton.title = 'Click to switch to horizontal';
        } else {
          icon.className = 'fas fa-desktop';
          text.textContent = 'Horizontal';
          toggleButton.title = 'Click to switch to vertical';
        }
      }
    }
    
    // Update camera constraints based on current orientation and device
    function updateCameraConstraints() {
      const cameraContainer = document.getElementById('cameraContainer');
      const deviceInfo = getDeviceOrientation();
      const isVerticalMode = cameraContainer.classList.contains('video-vertical');
      
      // Check if user has manually toggled (override auto-detection)
      const hasManualOverride = cameraContainer.dataset.manualOverride === 'true';
      
      // For tablets, use manual override if available, otherwise use device orientation
      if (deviceInfo.isTablet) {
        if (hasManualOverride) {
          // Use manual button state
          if (isVerticalMode) {
            window.cameraConstraints = {
              video: { 
                facingMode: { ideal: 'user' },
                width: { ideal: 720, max: 1080 },
                height: { ideal: 1280, max: 1920 },
                frameRate: { ideal: 15, max: 30 }
              } 
            };
            console.log('Tablet MANUAL VERTICAL constraints:', window.cameraConstraints);
          } else {
            window.cameraConstraints = {
              video: { 
                facingMode: { ideal: 'user' },
                width: { ideal: 1280, max: 1920 },
                height: { ideal: 720, max: 1080 },
                frameRate: { ideal: 15, max: 30 }
              } 
            };
            console.log('Tablet MANUAL HORIZONTAL constraints:', window.cameraConstraints);
          }
        } else {
          // Use auto-detection based on device orientation
          if (deviceInfo.isPortrait) {
            window.cameraConstraints = {
              video: { 
                facingMode: { ideal: 'user' },
                width: { ideal: 720, max: 1080 },
                height: { ideal: 1280, max: 1920 },
                frameRate: { ideal: 15, max: 30 }
              } 
            };
            console.log('Tablet AUTO portrait - VERTICAL constraints:', window.cameraConstraints);
          } else {
            window.cameraConstraints = {
              video: { 
                facingMode: { ideal: 'user' },
                width: { ideal: 1280, max: 1920 },
                height: { ideal: 720, max: 1080 },
                frameRate: { ideal: 15, max: 30 }
              } 
            };
            console.log('Tablet AUTO landscape - HORIZONTAL constraints:', window.cameraConstraints);
          }
        }
      }
      // For mobile devices, use button state
      else if (deviceInfo.isMobile) {
        if (isVerticalMode) {
          window.cameraConstraints = {
            video: { 
              facingMode: { ideal: 'user' },
              width: { ideal: 480, max: 720 },
              height: { ideal: 640, max: 1280 },
              frameRate: { ideal: 15, max: 30 }
            } 
          };
          console.log('Mobile VERTICAL constraints:', window.cameraConstraints);
        } else {
          window.cameraConstraints = {
            video: { 
              facingMode: { ideal: 'user' },
              width: { ideal: 640, max: 1280 },
              height: { ideal: 480, max: 720 },
              frameRate: { ideal: 15, max: 30 }
            } 
          };
          console.log('Mobile HORIZONTAL constraints:', window.cameraConstraints);
        }
      }
      // For desktop, use button state
      else {
        if (isVerticalMode) {
          window.cameraConstraints = {
            video: { 
              facingMode: { ideal: 'user' },
              width: { ideal: 720, max: 1080 },
              height: { ideal: 1280, max: 1920 },
              frameRate: { ideal: 15, max: 30 }
            } 
          };
          console.log('Desktop VERTICAL constraints:', window.cameraConstraints);
        } else {
          window.cameraConstraints = {
            video: { 
              facingMode: { ideal: 'user' },
              width: { ideal: 1280, max: 1920 },
              height: { ideal: 720, max: 1080 },
              frameRate: { ideal: 15, max: 30 }
            } 
          };
          console.log('Desktop HORIZONTAL constraints:', window.cameraConstraints);
        }
      }
    }

    async function startCam() {
      try {
        console.log('Requesting camera access...');
        // Use dynamic constraints based on current orientation
        updateCameraConstraints();
        console.log('Using camera constraints:', window.cameraConstraints);
        stream = await navigator.mediaDevices.getUserMedia(window.cameraConstraints);
        console.log('Camera stream obtained:', stream);
        
        video.srcObject = stream;
        console.log('Video srcObject set');
        
        // Wait for video to load
        video.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
          console.log('Video aspect ratio:', (video.videoWidth / video.videoHeight).toFixed(2));
          
          // Check if video orientation matches expected
          const deviceInfo = getDeviceOrientation();
          const isVerticalMode = document.getElementById('cameraContainer').classList.contains('video-vertical');
          const isPortraitVideo = video.videoHeight > video.videoWidth;
          
          console.log('Orientation check:', {
            devicePortrait: deviceInfo.isPortrait,
            containerVertical: isVerticalMode,
            videoPortrait: isPortraitVideo,
            matches: (deviceInfo.isPortrait && isPortraitVideo) || (!deviceInfo.isPortrait && !isPortraitVideo)
          });
          
          // Update canvas dimensions to match video
          if (canvas) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            console.log('Canvas updated to:', canvas.width, 'x', canvas.height);
          }
          
          // Show video and hide placeholder
          video.style.display = 'block';
          document.getElementById('cameraPlaceholder').style.display = 'none';
          
          // Update status
          cameraStatus.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; color: #28a745;"></i><span>Online</span>';
          cameraStatus.style.background = 'rgba(40, 167, 69, 0.9)';
          
          // Force sync overlay with video dimensions
          setTimeout(() => {
            forceSyncOverlay();
          }, 200);
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
        
        // Create canvas for image capture with mobile-optimized size
        canvas = document.createElement('canvas');
        // Use fallback dimensions if video not ready
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
        console.log('Canvas created with dimensions:', canvas.width, 'x', canvas.height);
        
        setButtons(true);
        setLog('Camera started.');
        updateDetectionDisplay(null, 'Camera active - detecting faces...');
        
        // Hide any existing countdown timer when camera starts
        hideCountdownTimer();
        
        // Reset successful scan flag when starting camera
        hasSuccessfulScan = false;
        
        // DIOPTIMALKAN: Start automatic face recognition every 500ms for faster response
        recognitionInterval = setInterval(performRecognition, 500);
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
      return new Promise((resolve) => {
        if (stream) {
          stream.getTracks().forEach(t => t.stop());
          video.srcObject = null;
          stream = null;
        }
        if (recognitionInterval) {
          clearInterval(recognitionInterval);
          recognitionInterval = null;
        }
        
        // Reset all processing flags when camera stops
        isProcessing = false;
        hasSuccessfulScan = false;
        remainingCooldown = 0;
        
        hideCountdownTimer(); // Hide countdown when camera stops
        setButtons(false);
        setLog('Camera stopped.');
        updateDetectionDisplay(null, 'Camera inactive');
        
        // Hide profile photo when camera stops
        hideMemberProfilePhoto();
        currentDisplayedMember = null;
        
        // Show placeholder and hide video
        video.style.display = 'none';
        document.getElementById('cameraPlaceholder').style.display = 'flex';
        
        // Update status
        cameraStatus.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; color: #dc3545;"></i><span>Offline</span>';
        cameraStatus.style.background = 'rgba(108, 117, 125, 0.9)';
        
        // Resolve after a small delay to ensure cleanup is complete
        setTimeout(resolve, 100);
      });
    }

    async function performRecognition() {
      if (!stream || !DOOR_TOKEN || !canvas || isProcessing) {
        console.log('Recognition skipped - stream:', !!stream, 'token:', !!DOOR_TOKEN, 'canvas:', !!canvas, 'processing:', isProcessing);
        return;
      }
      
      // Additional check to ensure we're not running after camera stop
      if (!stream) {
        console.log('Stream is null, stopping recognition');
        return;
      }
      
      // Check if we're in cooldown period - if so, don't scan at all
      if (remainingCooldown > 0) {
        console.log('In cooldown period, skipping recognition');
        return;
      }
      
      // Check if we already had a successful scan - prevent multiple scans
      if (hasSuccessfulScan) {
        console.log('Already had successful scan, skipping recognition - hasSuccessfulScan:', hasSuccessfulScan);
        return;
      }
      
      const now = Date.now();
      if (now - lastRecognitionTime < 500) {
        console.log('Throttling recognition - too soon since last scan');
        return;
      }
      
      console.log('Starting recognition process...');
      const startTime = Date.now();
      lastRecognitionTime = now;
      isProcessing = true; // Set flag to prevent multiple requests
      
      // Add basic face detection validation before sending to server
      // This helps reduce false positives when no face is actually visible
      // You can disable this check by setting ENABLE_BRIGHTNESS_CHECK to false
      const ENABLE_BRIGHTNESS_CHECK = false; // Set to true to enable brightness validation
      
      if (ENABLE_BRIGHTNESS_CHECK) {
        try {
          const ctx = canvas.getContext('2d');
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const data = imageData.data;
          
          // Simple brightness check - if image is too dark, likely no face
          let totalBrightness = 0;
          for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            totalBrightness += (r + g + b) / 3;
          }
          const avgBrightness = totalBrightness / (data.length / 4);
          
          // If image is too dark (likely no person), skip recognition
          // Reduced threshold to be less restrictive
          if (avgBrightness < 30) {
            console.log('Image too dark, likely no face present, skipping recognition');
            isProcessing = false;
            return;
          }
          
          console.log('Image brightness check passed:', avgBrightness);
        } catch (e) {
          console.log('Brightness check failed, proceeding with recognition:', e);
        }
      }
      
      try {
      const ctx = canvas.getContext('2d');
      
      // Ensure canvas has proper dimensions
      if (canvas.width === 0 || canvas.height === 0) {
        console.error('Canvas has zero dimensions');
        updateDetectionDisplay(null, 'Camera error - invalid dimensions');
        return;
      }
      
      // Ensure video is ready
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.error('Video not ready');
        updateDetectionDisplay(null, 'Camera error - video not ready');
        return;
      }
      
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 1.0);
      
      // Debug logging
      console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
      console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
      console.log('DataURL length:', dataUrl.length);
      console.log('Base64 length:', dataUrl.split(',')[1].length);
      
      // Validate image data
      if (dataUrl.length < 1000) {
        console.error('Image too small:', dataUrl.length);
        updateDetectionDisplay(null, 'Camera error - image too small');
        return;
      }
        
        // Check if we should reset noFaceCount based on client-side face detection
        // This is a simple check - if we're in no-face mode, try to reset it occasionally
        if (noFaceCount >= 3) {
          const now = Date.now();
          const timeSinceLastReset = now - lastNoFaceResetTime;
          
          // Check if image has changed significantly (camera condition changed)
          const currentImageData = canvas.toDataURL('image/jpeg', 0.1); // Low quality for comparison
          let imageChanged = false;
          
          if (lastImageData) {
            // Simple comparison - if images are very different, camera condition changed
            const similarity = calculateImageSimilarity(lastImageData, currentImageData);
            if (similarity < 0.7) { // Less than 70% similar
              imageChanged = true;
              console.log('Image changed significantly, similarity:', similarity);
            }
          }
          
          // Reset if image changed significantly OR if it's been at least 30 seconds since last reset
          if (imageChanged || (timeSinceLastReset > 30000 && Math.random() < 0.1)) {
            console.log('Resetting noFaceCount from', noFaceCount, 'to 0 due to', imageChanged ? 'image change' : 'timeout');
            noFaceCount = 0;
            lastNoFaceResetTime = now;
          } else {
            console.log('Skipping "Scanning..." display due to no-face mode (count:', noFaceCount, ', time since reset:', Math.round(timeSinceLastReset/1000), 's)');
            return;
          }
        }
        
        // Store current image data for next comparison
        lastImageData = canvas.toDataURL('image/jpeg', 0.1);
        
        // Show "Scanning..." display
        updateDetectionDisplay(null, 'Scanning...');
        console.log('Showing "Scanning..." display for recognition process');
        
      // DIOPTIMALKAN: Add timeout untuk mencegah hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      // DIOPTIMALKAN: Use fast recognition endpoint untuk speed
      const requestBody = {
        door_token: DOOR_TOKEN,
        image_b64: dataUrl.split(',')[1]
      };
      console.log('DEBUG: Sending request body:', {
        door_token: DOOR_TOKEN ? DOOR_TOKEN.substring(0, 20) + '...' : 'null',
        image_b64: 'present'
      });
      
      const r = await fetch('/api/recognize_fast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
        
        let j;
        try {
          j = await r.json();
          
          // Handle token expired - auto refresh
          if (r.status === 401 && j.error === "Invalid door token") {
            console.log("Token expired, refreshing...");
            await refreshDoorToken();
            return;
          }
          
          console.log('Server response:', j);
          console.log('Response details:', {
            ok: j.ok,
            matched: j.matched,
            success: j.success,
            name: j.name,
            candidate: j.candidate,
            gate: j.gate
          });
          
          // DIOPTIMALKAN: Enhanced debug logging untuk troubleshooting
          if (j.ok && j.matched) {
            console.log('✅ Face matched successfully');
            console.log('Name:', j.name || j.candidate?.email || 'No name');
            console.log('Success:', j.success);
            console.log('Gate:', j.gate);
            if (j.gate) {
              console.log('Gate popup_style:', j.gate.popup_style);
              console.log('Gate throttled:', j.gate.throttled);
            }
          } else {
            console.log('❌ Face not matched or error');
            console.log('Error:', j.error);
          }
        } catch (e) {
          console.error('JSON parse error:', e);
          updateDetectionDisplay(null, 'Server error: Invalid response');
          return;
        }
        
        // Handle server errors
        if (!r.ok) {
          console.error('Server error:', r.status, j);
          updateDetectionDisplay(null, j.error || 'Server error');
          return;
        }
        
        if (j.ok && j.matched && j.candidate) {
          const name = j.name || j.candidate.email || `Member ${j.candidate.gym_member_id}`;
          const confidence = j.best_score;
          
          // Mark face recognition as completed
          sessionStorage.setItem('face_registered', 'true');
          updateProgressRoadmap();
          
          // DIOPTIMALKAN: Handle success response dengan nama yang benar
          if (j.success && j.gate && j.gate.popup_style === 'GRANTED') {
            // Check-in berhasil
            updateDetectionDisplay(name, 'Check-in successful!', confidence, j.gate.popup_style);
            console.log('Check-in successful for:', name);
            
            // Tampilkan foto profil di pojok kanan bawah
            if (j.member_id) {
              showMemberProfilePhoto(j.member_id, name);
            }
          } else if (j.gate && j.gate.popup_style === 'GRANTED') {
            // Check-in berhasil (fallback untuk response tanpa success field)
            updateDetectionDisplay(name, 'Check-in successful!', confidence, j.gate.popup_style);
            console.log('Check-in successful for:', name);
            
            // Tampilkan foto profil di pojok kanan bawah
            if (j.member_id) {
              showMemberProfilePhoto(j.member_id, name);
            }
          } else if (j.gate && j.gate.throttled) {
            // User dalam cooldown
            updateDetectionDisplay(name, 'User in cooldown period', confidence);
            console.log('User in cooldown:', name);
            
            // Tampilkan foto profil di pojok kanan bawah untuk cooldown juga
            if (j.member_id) {
              showMemberProfilePhoto(j.member_id, name);
            }
          } else if (j.gate && (j.gate.popup_style === 'DENIED' || j.gate.response?.popup_style === 'DENIED')) {
            // Access denied
            const popupStyle = j.gate.popup_style || j.gate.response?.popup_style || 'DENIED';
            updateDetectionDisplay(name, 'Access Denied', confidence, popupStyle);
            console.log('Access denied for:', name);
          } else {
            // Face recognized but no gate action
            updateDetectionDisplay(name, 'Face recognized', confidence);
            console.log('Face recognized:', name);
          }
          
          // Draw face tracking for recognized face
          if (j.bbox && j.bbox.length === 4) {
            const [x1, y1, x2, y2] = j.bbox;
            console.log('Server bbox coordinates:', { x1, y1, x2, y2 });
            console.log('Calculated width/height:', { width: x2-x1, height: y2-y1 });
            
            // Apply mobile fullscreen offset to server coordinates
            const isMobileServer = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                                 window.innerWidth <= 768 || 
                                 ('ontouchstart' in window);
            const isFullscreenServer = document.getElementById('cameraContainer').classList.contains('fullscreen');
            
            let adjustedY1 = y1;
            let adjustedY2 = y2;
            
            if (isMobileServer) {
              let mobileServerOffset = 0;
              if (isFullscreenServer) {
                // Mobile fullscreen - moderate offset
                mobileServerOffset = video.videoHeight * 0.15; // 15% offset for mobile fullscreen
              } else {
                // Mobile normal - smaller offset
                mobileServerOffset = video.videoHeight * 0.05; // 5% offset for mobile normal
              }
              
              adjustedY1 = y1 + mobileServerOffset;
              adjustedY2 = y2 + mobileServerOffset;
              console.log('Applied mobile offset to server coordinates:', { originalY1: y1, adjustedY1, originalY2: y2, adjustedY2, isFullscreen: isFullscreenServer, offset: mobileServerOffset });
            }
            
            // Bounding box removed
          } else {
            // Use center-based fallback with device-specific positioning
            const centerX = video.videoWidth / 2;
            
            // Detect if device is mobile for fallback too
            const isMobileFallback = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                                   window.innerWidth <= 768 || 
                                   ('ontouchstart' in window);
            
            // Check if in fullscreen mode for additional offset
            const isFullscreen = document.getElementById('cameraContainer').classList.contains('fullscreen');
            
            // Special handling for mobile fullscreen - much larger offset
            let fullscreenOffset = 0;
            if (isFullscreen && isMobileFallback) {
              // Mobile fullscreen needs smaller offset
              fullscreenOffset = video.videoHeight * 0.10; // 10% of video height
            } else if (isFullscreen) {
              // Desktop fullscreen
              fullscreenOffset = video.videoHeight * 0.15;
            }
            
            // Different offset for mobile normal vs mobile fullscreen
            let mobileBaseOffset = 0;
            if (isMobileFallback && !isFullscreen) {
              // Mobile normal mode - smaller offset
              mobileBaseOffset = video.videoHeight * 0.05; // 5% for mobile normal
            } else if (isMobileFallback && isFullscreen) {
              // Mobile fullscreen - larger offset
              mobileBaseOffset = video.videoHeight * 0.20; // 20% for mobile fullscreen
            }
            
            const fallbackYOffset = isMobileFallback ? mobileBaseOffset + fullscreenOffset : (video.videoHeight * 0.05);
            const centerY = video.videoHeight / 2 + fallbackYOffset;
            
            // Use larger fallback size for better visibility on mobile
            const fallbackSize = Math.min(video.videoWidth, video.videoHeight) * 0.4;
            console.log('Using fallback coordinates:', { centerX, centerY, fallbackSize, isMobileFallback, fallbackYOffset });
            // Bounding box removed
          }
          
          // Check if user is throttled
          if (j.gate && j.gate.throttled) {
            console.log('User is throttled, showing center countdown animation');
            // Don't show popup, just show countdown animation
            updateDetectionDisplay(name, 'Cooldown active', confidence, null);
            
            // Get exact remaining time from server
            const serverCooldownSeconds = j.gate.cooldown_remaining || 10;
            console.log('Server cooldown seconds:', serverCooldownSeconds);
            
            // Use server time directly - server already calculated the correct remaining time
            const actualRemainingTime = Math.max(1, serverCooldownSeconds); // Ensure at least 1 second
            console.log('Using server cooldown time:', actualRemainingTime, 'seconds');
            
            // Show countdown timer with corrected time
            if (actualRemainingTime > 0) {
              showCountdownTimer(actualRemainingTime);
              
              // Stop recognition interval during cooldown
              if (recognitionInterval) {
                clearInterval(recognitionInterval);
                recognitionInterval = null;
                console.log('Recognition interval stopped due to cooldown');
              }
            } else {
              hideCountdownTimer();
            }
            
            // Change face tracking color to yellow for throttled user
            if (j.bbox && j.bbox.length === 4) {
              const [x1, y1, x2, y2] = j.bbox;
              console.log('Cooldown bbox coordinates:', { x1, y1, x2, y2 });
              
              // Apply mobile fullscreen offset to server coordinates for cooldown
              const isMobileCooldownServer = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                                           window.innerWidth <= 768 || 
                                           ('ontouchstart' in window);
              const isFullscreenCooldownServer = document.getElementById('cameraContainer').classList.contains('fullscreen');
              
              let adjustedCooldownY1 = y1;
              let adjustedCooldownY2 = y2;
              
              if (isMobileCooldownServer) {
                let mobileCooldownServerOffset = 0;
                if (isFullscreenCooldownServer) {
                  // Mobile fullscreen - moderate offset
                  mobileCooldownServerOffset = video.videoHeight * 0.15; // 15% offset for mobile fullscreen
                } else {
                  // Mobile normal - smaller offset
                  mobileCooldownServerOffset = video.videoHeight * 0.05; // 5% offset for mobile normal
                }
                
                adjustedCooldownY1 = y1 + mobileCooldownServerOffset;
                adjustedCooldownY2 = y2 + mobileCooldownServerOffset;
                console.log('Applied mobile offset to cooldown server coordinates:', { originalY1: y1, adjustedY1: adjustedCooldownY1, originalY2: y2, adjustedY2: adjustedCooldownY2, isFullscreen: isFullscreenCooldownServer, offset: mobileCooldownServerOffset });
              }
              
              // Bounding box removed
            } else {
              // Use center-based fallback for cooldown with device detection
              const centerX = video.videoWidth / 2;
              const isMobileCooldown = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                                     window.innerWidth <= 768 || 
                                     ('ontouchstart' in window);
              const isFullscreenCooldown = document.getElementById('cameraContainer').classList.contains('fullscreen');
              
              // Special handling for mobile fullscreen cooldown
              let fullscreenCooldownOffset = 0;
              if (isFullscreenCooldown && isMobileCooldown) {
                // Mobile fullscreen needs smaller offset
                fullscreenCooldownOffset = video.videoHeight * 0.10; // 10% of video height
              } else if (isFullscreenCooldown) {
                // Desktop fullscreen
                fullscreenCooldownOffset = video.videoHeight * 0.15;
              }
              
              // Different offset for mobile normal vs mobile fullscreen cooldown
              let mobileCooldownBaseOffset = 0;
              if (isMobileCooldown && !isFullscreenCooldown) {
                // Mobile normal mode - smaller offset
                mobileCooldownBaseOffset = video.videoHeight * 0.05; // 5% for mobile normal
              } else if (isMobileCooldown && isFullscreenCooldown) {
                // Mobile fullscreen - larger offset
                mobileCooldownBaseOffset = video.videoHeight * 0.20; // 20% for mobile fullscreen
              }
              
              const cooldownYOffset = isMobileCooldown ? mobileCooldownBaseOffset + fullscreenCooldownOffset : (video.videoHeight * 0.05);
              const centerY = video.videoHeight / 2 + cooldownYOffset;
              const fallbackSize = Math.min(video.videoWidth, video.videoHeight) * 0.4;
              console.log('Using cooldown fallback coordinates:', { centerX, centerY, fallbackSize, isMobileCooldown, cooldownYOffset });
              // Bounding box removed
            }
          } else if (j.gate && !j.gate.error) {
            // Auto-open gate if matched and not throttled
            console.log('Gate opened successfully, hiding cooldown timer');
            const popupStyle = j.gate.popup_style || j.gate.response?.popup_style || j.popup_style || 'GRANTED';
            
            // Display appropriate message based on popup_style
            let gateMessage = 'Gate opened successfully!';
            if (popupStyle === 'DENIED') {
              gateMessage = 'Access Denied';
            } else if (popupStyle === 'WARNING') {
              gateMessage = 'Gate opened with warning';
            }
            
            updateDetectionDisplay(name, gateMessage, confidence, popupStyle);
            // Don't hide countdown timer here - we want to show cooldown after successful scan
            
            // Show profile photo immediately after successful recognition
            if (j.member_id) {
              console.log('Showing profile photo for successful recognition:', j.member_id, name);
              console.log('Calling showMemberProfilePhoto with member_id:', j.member_id, 'name:', name);
              await showMemberProfilePhoto(j.member_id, name);
              console.log('Profile photo should be visible now');
            } else {
              console.log('No member_id in response, cannot show profile photo');
              console.log('Response data:', j);
            }
            
            // Stop recognition interval immediately after successful scan to prevent multiple scans
            if (recognitionInterval) {
              clearInterval(recognitionInterval);
              recognitionInterval = null;
              console.log('Recognition interval stopped after successful scan');
            }
            
            // Also set isProcessing to false to prevent any pending requests
            isProcessing = false;
            hasSuccessfulScan = true; // Mark that we had a successful scan
            console.log('Processing flag reset after successful scan');
            
            // Reset no face counter on successful recognition
            if (noFaceCount > 0) {
              console.log('Resetting noFaceCount from', noFaceCount, 'to 0 on successful recognition');
              noFaceCount = 0;
              lastNoFaceResetTime = Date.now();
            }
            
            // Start cooldown after successful scan
            console.log('Starting cooldown after successful scan');
            remainingCooldown = 5; // Set 10 seconds cooldown
            
            // Check if we're in fullscreen mode for enhanced countdown
            const isFullscreen = document.getElementById('cameraContainer').classList.contains('fullscreen');
            console.log('Starting countdown in fullscreen mode:', isFullscreen);
            
            // Add a small delay to ensure popup is shown first
            setTimeout(() => {
              showCountdownTimer(remainingCooldown);
              
            // Log countdown start for debugging
            console.log('Countdown timer started with', remainingCooldown, 'seconds remaining');
            
            // Show success message in fullscreen mode
            if (isFullscreen) {
              console.log('Fullscreen countdown started - enhanced UI will be shown');
              
              // Add visual feedback for fullscreen countdown start
              const centerCountdown = document.getElementById('centerCountdown');
              if (centerCountdown) {
                centerCountdown.style.border = '3px solid rgba(76, 175, 80, 0.8)';
                centerCountdown.style.boxShadow = '0 0 30px rgba(76, 175, 80, 0.3), 0 20px 60px rgba(0,0,0,0.8)';
                
                // Reset border after countdown starts
                setTimeout(() => {
                  centerCountdown.style.border = '';
                  centerCountdown.style.boxShadow = '';
                }, 2000);
              }
            }
            }, 500);
            
            // Record successful recognition time for cooldown calculation
            lastSuccessfulRecognitionTime = Date.now();
            console.log('Successful recognition recorded at:', new Date(lastSuccessfulRecognitionTime));
          }
        } else if (j.ok && !j.matched) {
          // Check if it's a "no face" error - don't show popup for these
          const isNoFaceError = j.error && (
            j.error.includes('No face detected') || 
            j.error.includes('No valid face detected') ||
            j.error.includes('face too small')
          );
          
          if (isNoFaceError) {
            console.log('No face detected in matched=false case, updating display quietly:', j.error);
            updateDetectionDisplay(null, j.error, null, null);
            
            // Hide loading overlay for no face detected
            const loadingOverlay = document.getElementById('loadingOverlay');
            if (loadingOverlay) {
              loadingOverlay.classList.remove('show');
            }
            
            // Increment no face counter and reduce scanning frequency
            noFaceCount++;
            console.log('No face detected in matched=false case, count:', noFaceCount);
            
            // If no face detected multiple times, reduce scanning frequency
            if (noFaceCount >= 3) {
              console.log('No face detected multiple times in matched=false case, reducing scanning frequency');
              // Clear current interval and restart with longer interval
              if (recognitionInterval) {
                clearInterval(recognitionInterval);
                recognitionInterval = setInterval(performRecognition, 2000); // 2 seconds instead of 500ms
                console.log('Reduced scanning frequency to 2 seconds');
              }
            }
          } else {
            // Face detected but not recognized - reset no face counter
            console.log('Face detected but not recognized, resetting noFaceCount from', noFaceCount, 'to 0');
            noFaceCount = 0;
            lastNoFaceResetTime = Date.now();
            
            // Only show popup for actual unknown faces (not no face detected)
            const popupStyle = j.popup_style || 'DENIED';
            updateDetectionDisplay('Wajah Belum Terdaftar Harap Retake', 'Face detected but not recognized', null, popupStyle);
          }
          
          // Hide profile photo for unknown face
          hideMemberProfilePhoto();
          currentDisplayedMember = null;
          // Draw orange face tracking for unknown face
          if (j.bbox && j.bbox.length === 4) {
            const [x1, y1, x2, y2] = j.bbox;
            console.log('Unknown face bbox coordinates:', { x1, y1, x2, y2 });
            
            // Apply mobile fullscreen offset to server coordinates for unknown
            const isMobileUnknownServer = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                                       window.innerWidth <= 768 || 
                                       ('ontouchstart' in window);
            const isFullscreenUnknownServer = document.getElementById('cameraContainer').classList.contains('fullscreen');
            
            let adjustedUnknownY1 = y1;
            let adjustedUnknownY2 = y2;
            
            if (isMobileUnknownServer) {
              let mobileUnknownServerOffset = 0;
              if (isFullscreenUnknownServer) {
                // Mobile fullscreen - moderate offset
                mobileUnknownServerOffset = video.videoHeight * 0.15; // 15% offset for mobile fullscreen
              } else {
                // Mobile normal - smaller offset
                mobileUnknownServerOffset = video.videoHeight * 0.05; // 5% offset for mobile normal
              }
              
              adjustedUnknownY1 = y1 + mobileUnknownServerOffset;
              adjustedUnknownY2 = y2 + mobileUnknownServerOffset;
              console.log('Applied mobile offset to unknown server coordinates:', { originalY1: y1, adjustedY1: adjustedUnknownY1, originalY2: y2, adjustedY2: adjustedUnknownY2, isFullscreen: isFullscreenUnknownServer, offset: mobileUnknownServerOffset });
            }
            
            // Bounding box removed
          } else {
            // Use center-based fallback for unknown with device detection
            const centerX = video.videoWidth / 2;
            const isMobileUnknown = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                                  window.innerWidth <= 768 || 
                                  ('ontouchstart' in window);
            const isFullscreenUnknown = document.getElementById('cameraContainer').classList.contains('fullscreen');
            
            // Special handling for mobile fullscreen unknown
            let fullscreenUnknownOffset = 0;
            if (isFullscreenUnknown && isMobileUnknown) {
              // Mobile fullscreen needs smaller offset
              fullscreenUnknownOffset = video.videoHeight * 0.10; // 10% of video height
            } else if (isFullscreenUnknown) {
              // Desktop fullscreen
              fullscreenUnknownOffset = video.videoHeight * 0.15;
            }
            
            // Different offset for mobile normal vs mobile fullscreen unknown
            let mobileUnknownBaseOffset = 0;
            if (isMobileUnknown && !isFullscreenUnknown) {
              // Mobile normal mode - smaller offset
              mobileUnknownBaseOffset = video.videoHeight * 0.05; // 5% for mobile normal
            } else if (isMobileUnknown && isFullscreenUnknown) {
              // Mobile fullscreen - larger offset
              mobileUnknownBaseOffset = video.videoHeight * 0.05; // 20% for mobile fullscreen
            }
            
            const unknownYOffset = isMobileUnknown ? mobileUnknownBaseOffset + fullscreenUnknownOffset : (video.videoHeight * 0.05);
            const centerY = video.videoHeight / 2 + unknownYOffset;
            const fallbackSize = Math.min(video.videoWidth, video.videoHeight) * 0.4;
            console.log('Using unknown fallback coordinates:', { centerX, centerY, fallbackSize, isMobileUnknown, unknownYOffset });
            // Bounding box removed
          }
        } else {
          // Don't show popup for "No face detected" or "No valid face detected" - just update display quietly
          const errorMessage = j.error || 'No face detected';
          console.log('No face detected, updating display quietly:', errorMessage);
          
          // Check if it's a "no face" or "no valid face" error - don't show popup for these
          const isNoFaceError = errorMessage.includes('No face detected') || 
                               errorMessage.includes('No valid face detected') ||
                               errorMessage.includes('face too small');
          
          if (isNoFaceError) {
            // Update display without popup for no face detected
            updateDetectionDisplay(null, errorMessage, null, null);
            
            // Hide loading overlay for no face detected
            const loadingOverlay = document.getElementById('loadingOverlay');
            if (loadingOverlay) {
              loadingOverlay.classList.remove('show');
            }
            
            // Increment no face counter and reduce scanning frequency
            noFaceCount++;
            console.log('No face detected, count:', noFaceCount);
            
            // If no face detected multiple times, reduce scanning frequency
            if (noFaceCount >= 3) {
              console.log('No face detected multiple times, reducing scanning frequency');
              // Clear current interval and restart with longer interval
              if (recognitionInterval) {
                clearInterval(recognitionInterval);
                recognitionInterval = setInterval(performRecognition, 2000); // 2 seconds instead of 500ms
                console.log('Reduced scanning frequency to 2 seconds');
              }
            }
          } else {
            // Check if this is a "no face detected" error or other error
            const isNoFaceError = errorMessage.includes('No face detected') || 
                                  errorMessage.includes('No valid face detected') || 
                                  errorMessage.includes('face too small');
            
            if (isNoFaceError) {
              // No face detected - don't reset counter, just update display
              console.log('No face detected error, keeping noFaceCount at', noFaceCount);
              updateDetectionDisplay(null, errorMessage, null, null); // No popup for no face
            } else {
              // Face detected but other error - reset no face counter
              console.log('Face detected but other error, resetting noFaceCount from', noFaceCount, 'to 0');
              noFaceCount = 0;
              lastNoFaceResetTime = Date.now();
              
              // Show popup for other errors
              updateDetectionDisplay(null, errorMessage, null, 'DENIED');
            }
          }
          clearFaceIndicator();
          
          // Hide profile photo when no face detected
          hideMemberProfilePhoto();
          currentDisplayedMember = null;
        }
        } catch (e) {
        const endTime = Date.now();
        const processingTime = endTime - startTime;
        console.log(`Recognition processing time: ${processingTime}ms`);
        
        if (e.name === 'AbortError') {
          updateDetectionDisplay(null, 'Recognition timeout - please try again');
          setLog('Recognition timeout - please try again');
        } else {
          updateDetectionDisplay(null, 'Recognition error: ' + e.message);
          setLog('Recognition error: ' + e.message);
        }
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
      console.log('Stopping camera and clearing recognition interval');
      stopCam().then(() => {
        console.log('Camera stopped successfully');
        console.log('Recognition interval cleared:', !recognitionInterval);
        console.log('Stream cleared:', !stream);
      });
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
    
    // Add keyboard shortcut for F11 (like browser fullscreen)
    document.addEventListener('keydown', (e) => {
      if (e.key === 'F11') {
        e.preventDefault();
        toggleFullscreen();
      }
    });
    
    // Handle window resize for mobile responsiveness
    window.addEventListener('resize', () => {
      if (overlay && video) {
        // Force sync overlay with video when window resizes
        setTimeout(() => {
          forceSyncOverlay();
        }, 200); // Longer delay for mobile
      }
    });
    
    // Handle orientation change for mobile
    window.addEventListener('orientationchange', () => {
      setTimeout(() => {
        if (overlay && video) {
          // Force sync overlay with video after orientation change
          forceSyncOverlay();
        }
        // Restart camera after orientation change for better mobile compatibility
        if (stream && video.videoWidth === 0) {
          console.log('Restarting camera after orientation change');
          stopCam();
          setTimeout(() => startCam(), 1000);
        }
        
        // Additional sync for mobile screens
        setTimeout(() => {
          forceSyncOverlay();
        }, 500);
      }, 100);
    });
    
    // Initialize progress roadmap on page load
    document.addEventListener('DOMContentLoaded', function() {
      console.log('Initializing progress roadmap...');
      updateProgressRoadmap();
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
      
      // Test if we can access camera with mobile-optimized constraints
      navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 480 },
          height: { ideal: 640 },
          frameRate: { ideal: 15 }
        } 
      })
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
        btn.style.background = '#0072bc';
      } else {
        overlay.style.display = 'none';
        btn.textContent = 'Show Overlay';
        btn.style.background = '#28a745';
      }
    }
    
    // Global variable to track current displayed member
    let currentDisplayedMember = null;
    
    
    // Function to show profile placeholder
    function showMemberProfilePlaceholder(memberName) {
      const container = document.getElementById('profilePhotoContainer');
      const placeholder = document.getElementById('profilePhotoPlaceholder');
      const photo = document.getElementById('profilePhoto');
      const nameOverlay = document.getElementById('profileNameOverlay');
      
      // Hide photo and show placeholder
      photo.style.display = 'none';
      placeholder.style.display = 'flex';
      
      // Set member name
      nameOverlay.textContent = memberName;
      
      // Show container with animation
      container.classList.add('show');
      container.style.display = 'block';
      
      // Keep placeholder visible - no auto-hide
    }
    
    // Function to hide member profile photo
    function hideMemberProfilePhoto() {
      const container = document.getElementById('profilePhotoContainer');
      container.classList.remove('show');
      
      // Hide after animation completes
      setTimeout(() => {
        container.style.display = 'none';
      }, 500);
    }
    
    // Function to show member profile photo (renamed from showMemberProfilePhoto)
    async function showMemberProfilePhoto(memberId, memberName) {
      try {
        // If different member is detected, hide current photo first
        if (currentDisplayedMember !== null && currentDisplayedMember !== memberId) {
          console.log('Different member detected, hiding current photo');
          hideMemberProfilePhoto();
        }
        
        // Always show photo for recognized member, even if same member
        console.log('Showing profile photo for member:', memberId);
        
        // Always show photo, even for same member
        console.log('Always showing photo for member:', memberId);
        
        console.log('Fetching profile photo for member:', memberId);
        currentDisplayedMember = memberId;
        
        console.log('Making API call to /api/get_member_photo/' + memberId);
        const response = await fetch(`/api/get_member_photo/${memberId}`);
        console.log('API response status:', response.status);
        const data = await response.json();
        console.log('API response data:', data);
        
        if (data.ok && data.photo_url) {
          console.log('Profile photo data is valid, setting up DOM elements');
          const container = document.getElementById('profilePhotoContainer');
          const placeholder = document.getElementById('profilePhotoPlaceholder');
          const photo = document.getElementById('profilePhoto');
          const nameOverlay = document.getElementById('profileNameOverlay');
          
          console.log('DOM elements found:', {
            container: !!container,
            placeholder: !!placeholder,
            photo: !!photo,
            nameOverlay: !!nameOverlay
          });
          
          // Set photo source
          const photoUrl = data.photo_url + (data.photo_url.includes('?') ? '&' : '?') + 't=' + Date.now();
          console.log('Setting photo source to:', photoUrl);
          photo.src = photoUrl;
          
          // Set member name
          const memberName = data.full_name || memberName || `Member ${memberId}`;
          console.log('Setting member name to:', memberName);
          nameOverlay.textContent = memberName;
          
          // Show photo and hide placeholder
          console.log('Showing photo and hiding placeholder');
          photo.style.display = 'block';
          photo.style.visibility = 'visible';
          photo.style.opacity = '1';
          placeholder.style.display = 'none';
          
          // Show container with animation
          console.log('Adding show class and setting display block');
          container.classList.add('show');
          container.style.display = 'block';
          
          // Force visibility with important styles
          container.style.visibility = 'visible';
          container.style.opacity = '1';
          container.style.zIndex = '1000';
          
          console.log('Profile photo displayed for:', data.full_name);
          console.log('Container final state:', {
            display: container.style.display,
            classList: container.classList.toString(),
            photoDisplay: photo.style.display,
            placeholderDisplay: placeholder.style.display
          });
          
          // Keep photo visible - no auto-hide
          
          // Fallback: Force show after a short delay
          setTimeout(() => {
            if (container.style.display === 'none' || !container.classList.contains('show')) {
              console.log('Fallback: Forcing container to show');
              container.style.display = 'block';
              container.style.visibility = 'visible';
              container.style.opacity = '1';
              container.classList.add('show');
            }
          }, 100);
          
        } else {
          console.log('No profile photo available for member:', memberId);
          console.log('API response was not ok or no photo_url:', data);
          // Show placeholder with member name
          showMemberProfilePlaceholder(memberName || `Member ${memberId}`);
        }
        
      } catch (error) {
        console.error('Error fetching profile photo:', error);
        // Show placeholder with member name
        showMemberProfilePlaceholder(memberName || `Member ${memberId}`);
      }
    }
    

    setButtons(false);
    if (DOOR_TOKEN) {
      btnStart.disabled = false;
    }
    
    // Initialize orientation toggle button
    document.addEventListener('DOMContentLoaded', function() {
      const toggleButton = document.getElementById('orientationToggle');
      if (toggleButton) {
        toggleButton.classList.add('vertical');
        const icon = toggleButton.querySelector('i');
        const text = toggleButton.querySelector('span');
        icon.className = 'fas fa-mobile-alt';
        text.textContent = 'Vertical';
      }
      
      // Initialize camera container classes based on device orientation
      const cameraContainer = document.getElementById('cameraContainer');
      const deviceInfo = getDeviceOrientation();
      
      if (deviceInfo.isTablet) {
        // For tablets, set initial classes based on device orientation
        if (deviceInfo.isPortrait) {
          cameraContainer.classList.remove('video-horizontal');
          cameraContainer.classList.add('video-vertical');
          console.log('Initialized tablet container to vertical for portrait device');
        } else {
          cameraContainer.classList.remove('video-vertical');
          cameraContainer.classList.add('video-horizontal');
          console.log('Initialized tablet container to horizontal for landscape device');
        }
        // Ensure manual override is false initially
        cameraContainer.dataset.manualOverride = 'false';
      }
      
      // Initialize camera constraints
      updateCameraConstraints();
      
      // Update button text based on device orientation
      updateOrientationButton();
      
      // Add orientation change listener for tablets
      let lastOrientation = window.innerHeight > window.innerWidth;
      window.addEventListener('resize', function() {
        const currentOrientation = window.innerHeight > window.innerWidth;
        const deviceInfo = getDeviceOrientation();
        
        // Update button text for all devices
        updateOrientationButton();
        
        // Only restart camera if orientation changed and it's a tablet
        if (deviceInfo.isTablet && currentOrientation !== lastOrientation) {
          console.log('Tablet orientation changed, restarting camera...');
          
          // Reset manual override when device orientation changes
          const cameraContainer = document.getElementById('cameraContainer');
          cameraContainer.dataset.manualOverride = 'false';
          console.log('Reset manual override due to device orientation change');
          
          // Update camera container classes to match device orientation
          if (currentOrientation) {
            // Device is in portrait - use vertical
            cameraContainer.classList.remove('video-horizontal');
            cameraContainer.classList.add('video-vertical');
            console.log('Updated container to vertical for portrait device');
          } else {
            // Device is in landscape - use horizontal
            cameraContainer.classList.remove('video-vertical');
            cameraContainer.classList.add('video-horizontal');
            console.log('Updated container to horizontal for landscape device');
          }
          
          // Update camera constraints
          updateCameraConstraints();
          
          const video = document.getElementById('video');
          const isCameraActive = video && video.srcObject && !video.paused;
          
          if (isCameraActive) {
            stopCam().then(() => {
              setTimeout(() => {
                startCam();
              }, 500);
            });
          }
        }
        
        lastOrientation = currentOrientation;
      });
    });
  </script>
  
  <!-- Footer -->
  <footer>
    <p>© <span id="currentYear"></span> FTL IT Developer. All rights reserved.</p>
  </footer>
  
  <!-- Dynamic Year Script -->
  <script>
    // Set current year dynamically
    document.getElementById('currentYear').textContent = new Date().getFullYear();
  </script>

  <!-- WebSocket Client for Real-time Control -->
  <script>
    (function() {
      // Initialize Socket.IO connection
      const socket = io();
      let reconnectAttempts = 0;
      const maxReconnectAttempts = 10;
      
      console.log('🔌 Initializing WebSocket connection...');
      
      // Connection established
      socket.on('connect', function() {
        console.log('✅ WebSocket connected!');
        reconnectAttempts = 0;
        
        // Register device with server
        socket.emit('register_device', {
          device_name: navigator.userAgent.includes('Mobile') ? 'Tablet Device' : 'Desktop Device',
          device_type: navigator.userAgent.includes('Mobile') ? 'Tablet' : 'Desktop',
          location: window.location.hostname,
          doorid: '{{ doorid }}' || null
        });
        
        // Start heartbeat
        setInterval(function() {
          socket.emit('heartbeat', { timestamp: new Date().toISOString() });
        }, 30000); // Every 30 seconds
      });
      
      // Connection status
      socket.on('connection_status', function(data) {
        console.log('📱 Connection Status:', data);
      });
      
      // Registration confirmation
      socket.on('registration_success', function(data) {
        console.log('✅ Device registered:', data);
      });
      
      // Force refresh command from server
      socket.on('force_refresh', function(data) {
        console.log('🔄 Received refresh command from server:', data.message);
        
        // Show notification before refresh
        const notification = document.createElement('div');
        notification.style.cssText = `
          position: fixed;
          top: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
          color: white;
          padding: 15px 30px;
          border-radius: 10px;
          box-shadow: 0 5px 20px rgba(0,0,0,0.3);
          z-index: 10000;
          font-weight: 600;
          text-align: center;
          animation: slideDown 0.3s ease-out;
        `;
        notification.innerHTML = `
          <i class="fas fa-sync-alt"></i> ${data.message || 'Refreshing page...'}
        `;
        document.body.appendChild(notification);
        
        // Refresh page after 1.5 seconds
        setTimeout(function() {
          window.location.reload();
        }, 1500);
      });
      
      // Server notification
      socket.on('notification', function(data) {
        console.log('📢 Notification:', data);
        
        const colors = {
          info: '#3b82f6',
          success: '#10b981',
          warning: '#f59e0b',
          error: '#ef4444'
        };
        
        const notification = document.createElement('div');
        notification.style.cssText = `
          position: fixed;
          top: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: ${colors[data.type] || colors.info};
          color: white;
          padding: 15px 30px;
          border-radius: 10px;
          box-shadow: 0 5px 20px rgba(0,0,0,0.3);
          z-index: 10000;
          font-weight: 600;
          text-align: center;
          animation: slideDown 0.3s ease-out;
        `;
        notification.innerHTML = `<i class="fas fa-bell"></i> ${data.message}`;
        document.body.appendChild(notification);
        
        setTimeout(function() {
          notification.style.animation = 'slideUp 0.3s ease-out';
          setTimeout(function() {
            notification.remove();
          }, 300);
        }, 5000);
      });
      
      // Device count update
      socket.on('device_count', function(data) {
        console.log('📊 Connected devices:', data.count);
      });
      
      // Heartbeat acknowledgment
      socket.on('heartbeat_ack', function(data) {
        console.log('💓 Heartbeat acknowledged');
      });
      
      // Disconnection handling
      socket.on('disconnect', function() {
        console.log('❌ WebSocket disconnected');
        reconnectAttempts++;
        
        if (reconnectAttempts <= maxReconnectAttempts) {
          console.log(`🔄 Reconnecting... (Attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
        }
      });
      
      // Connection error
      socket.on('connect_error', function(error) {
        console.error('❌ Connection error:', error);
      });
      
      // Add CSS animations
      const style = document.createElement('style');
      style.textContent = `
        @keyframes slideDown {
          from { transform: translate(-50%, -100px); opacity: 0; }
          to { transform: translate(-50%, 0); opacity: 1; }
        }
        @keyframes slideUp {
          from { transform: translate(-50%, 0); opacity: 1; }
          to { transform: translate(-50%, -100px); opacity: 0; }
        }
      `;
      document.head.appendChild(style);
      
      console.log('✅ WebSocket client initialized');
    })();
  </script>
</body>
</html>
"""
# Ini untuk main page Face Recognition END

# Setelah Login pasti masuk sini START



@app.route("/", methods=["GET", "POST"])
def index():
    doorid = None
    
    if request.method == "POST":
        # Handle POST request with doorid in JSON body
        data = request.get_json(force=True)
        doorid = data.get("doorid") if data else None
        
        # If no JSON data, try form data
        if not doorid:
            doorid = request.form.get("doorid")
        
        # Store doorid in session
        if doorid:
            session['selected_doorid'] = doorid
    
    elif request.method == "GET":
        # Handle GET request with doorid in query parameters
        doorid = request.args.get("doorid")
        
        # Store doorid in session if provided
        if doorid:
            session['selected_doorid'] = doorid
    
    # Use doorid from session if no doorid in request
    if not doorid:
        doorid = session.get('selected_doorid')
    
    # Allow access to main page without requiring doorid or admin login
    # If doorid is present, issue door token
    if doorid:
        try:
            device_fp = _client_fingerprint(request)
            token = sign_door_token(int(doorid), device_fp)
            print(f"DEBUG: Created door token for doorid {doorid}: {token[:20]}...")
            return render_template_string(INDEX_HTML, doorid=doorid, door_token=token)
        except Exception as e:
            print(f"DEBUG: Error creating door token: {e}")
            return render_template_string(INDEX_HTML, doorid=doorid, door_token="")
    
    # Render main page without doorid (public access)
    return render_template_string(INDEX_HTML, doorid=None, door_token="")


@app.route("/api/access_gate", methods=["POST"])
def api_access_gate():
    """API endpoint untuk POST request dengan doorid"""
    data = request.get_json(force=True)
    doorid = data.get("doorid")
    
    if not doorid:
        return jsonify({"ok": False, "error": "doorid is required"}), 400
    
    # Store doorid in session
    session['selected_doorid'] = doorid
    
    # Return JSON response with redirect URL (no doorid in URL)
    return jsonify({
        "ok": True, 
        "redirect_url": "/",  # Clean URL without parameters
        "doorid": doorid
    })

@app.route("/pantooo")
def door_select():
    """Redirect to admin login for door selection"""
    return redirect(url_for('admin_login'))

@app.route("/admin/door_select")
@require_admin_auth
def admin_door_select():
    """Halaman untuk memilih doorid setelah login admin"""
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>FTL Face Gate - Select Door</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { 
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 400px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #333;
        }
        input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
        }
        .btn {
            width: 100%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn:hover {
            background: #0056b3;
        }
        .logout-btn {
            background: #dc3545;
            margin-top: 10px;
        }
        .logout-btn:hover {
            background: #c82333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 style="text-align: center; margin-bottom: 30px; color: #333;">
            <i class="fas fa-door-open"></i> Select Door
        </h2>
        
        <form id="doorForm" method="POST" action="/api/access_gate">
            <div class="form-group">
                <label for="doorid">Door ID:</label>
                <input type="number" id="doorid" name="doorid" placeholder="Enter Door ID" required>
            </div>
            
            <button type="submit" class="btn">
                <i class="fas fa-arrow-right"></i> Access Gate
            </button>
            
            <button type="button" class="btn logout-btn" onclick="window.location.href='/admin/logout'">
                <i class="fas fa-sign-out-alt"></i> Logout
            </button>
        </form>
    </div>

    <script>
        // Form submission
        document.getElementById('doorForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const doorid = document.getElementById('doorid').value;
            if (!doorid) {
                alert('Please enter a door ID');
                return;
            }
            
            // Submit form
            fetch('/api/access_gate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({doorid: doorid})
            })
            .then(response => response.json())
            .then(data => {
                if (data.ok) {
                    // Redirect to main page without door ID in URL (security)
                    window.location.href = data.redirect_url;
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Network error: ' + error.message);
            });
        });
    </script>
</body>
</html>
    """)



@app.route("/admin/login", methods=["GET", "POST"])
@limiter.limit(LOGIN_RATE_LIMIT)
def admin_login():
    if request.method == "POST":
        client_id = get_client_identifier()
        
        # Check if login is blocked due to brute force attempts
        if is_login_blocked(client_id):
            remaining_time = get_remaining_lockout_time(client_id)
            error_msg = f"Too many failed login attempts. Please try again in {remaining_time} seconds."
            return render_template_string(ADMIN_LOGIN_HTML, error=error_msg)
        
        username = request.form.get("username")
        password = request.form.get("password")
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            # Reset login attempts on successful login
            reset_login_attempts(client_id)
            session['admin_authenticated'] = True
            return redirect(url_for('admin_door_select'))
        else:
            # Increment failed login attempts
            attempts = increment_login_attempts(client_id)
            remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts
            
            if remaining_attempts <= 0:
                remaining_time = get_remaining_lockout_time(client_id)
                error_msg = f"Too many failed login attempts. Please try again in {remaining_time} seconds."
            else:
                error_msg = f"Invalid credentials. {remaining_attempts} attempts remaining."
            
            return render_template_string(ADMIN_LOGIN_HTML, error=error_msg)
    
    return render_template_string(ADMIN_LOGIN_HTML)

@app.route("/admin/logout")
def admin_logout():
    session.pop('admin_authenticated', None)
    return redirect(url_for('admin_login'))

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {}
    }
    
    # Check database connection
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        health_status["services"]["database"] = {
            "status": "healthy",
            "host": DB["host"],
            "port": DB["port"],
            "database": DB["database"]
        }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis connection
    try:
        redis_conn = get_redis_conn()
        if redis_conn:
            redis_conn.ping()
            health_status["services"]["redis"] = {
                "status": "healthy",
                "host": REDIS_HOST,
                "port": REDIS_PORT,
                "database": REDIS_DB
            }
        else:
            health_status["services"]["redis"] = {
                "status": "unavailable",
                "message": "Redis connection failed"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check GymMaster API
    try:
        response = requests.get(f"{GYM_BASE_URL}/health", timeout=5)
        health_status["services"]["gymmaster_api"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "base_url": GYM_BASE_URL,
            "status_code": response.status_code
        }
    except Exception as e:
        health_status["services"]["gymmaster_api"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Return appropriate HTTP status
    if health_status["status"] == "healthy":
        return jsonify(health_status), 200
    elif health_status["status"] == "degraded":
        return jsonify(health_status), 200  # Still return 200 but with degraded status
    else:
        return jsonify(health_status), 503

@app.route("/api/security/status", methods=["GET"])
@require_admin_auth
def api_security_status():
    """Get security status including rate limiting and brute force protection info"""
    client_id = get_client_identifier()
    
    # Get current rate limiting status
    redis_conn = get_redis_conn()
    rate_limit_info = {}
    
    if redis_conn:
        try:
            # Get rate limiting keys
            rate_limit_keys = redis_conn.keys("LIMITER:*")
            rate_limit_info["active_rate_limits"] = len(rate_limit_keys)
            
            # Get login attempt info
            login_key = f"login_attempts:{client_id}"
            login_attempts = redis_conn.get(login_key)
            if login_attempts:
                rate_limit_info["current_login_attempts"] = int(login_attempts)
                rate_limit_info["remaining_lockout_time"] = get_remaining_lockout_time(client_id)
            else:
                rate_limit_info["current_login_attempts"] = 0
                rate_limit_info["remaining_lockout_time"] = 0
                
        except Exception as e:
            rate_limit_info["error"] = str(e)
    
    # Get branch information
    branch_id = get_branch_identifier()
    branch_capacity_info = {}
    
    if redis_conn:
        try:
            branch_key = f"branch_users:{branch_id}"
            current_users = redis_conn.scard(branch_key)
            branch_capacity_info = {
                "current_users": current_users,
                "max_users": MAX_USERS_PER_BRANCH,
                "capacity_remaining": MAX_USERS_PER_BRANCH - current_users
            }
        except Exception as e:
            branch_capacity_info = {"error": str(e)}
    
    return jsonify({
        "ok": True,
        "security_status": {
            "client_identifier": client_id,
            "branch_identifier": branch_id,
            "max_login_attempts": MAX_LOGIN_ATTEMPTS,
            "login_lockout_time": LOGIN_LOCKOUT_TIME,
            "api_rate_limit": API_RATE_LIMIT,
            "login_rate_limit": LOGIN_RATE_LIMIT,
            "user_rate_limit": USER_RATE_LIMIT,
            "branch_rate_limit": BRANCH_RATE_LIMIT,
            "max_users_per_branch": MAX_USERS_PER_BRANCH,
            "rate_limit_info": rate_limit_info,
            "branch_capacity": branch_capacity_info,
            "timestamp": datetime.now().isoformat()
        }
    })


# -------------------- Image & Recognition Utils --------------------

def b64_to_bgr(image_b64: str) -> Optional[np.ndarray]:
    try:
        if not image_b64 or len(image_b64.strip()) == 0:
            print("DEBUG: Empty image_b64")
            return None
            
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]
            
        # Validate base64 string
        if len(image_b64) < 100:  # Too small for a valid image
            print(f"DEBUG: Image too small: {len(image_b64)} characters")
            return None
            
        data = base64.b64decode(image_b64)
        if len(data) < 1000:  # Too small for a valid image
            print(f"DEBUG: Decoded data too small: {len(data)} bytes")
            return None
            
        im = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(im, cv2.IMREAD_COLOR)
        
        if bgr is None:
            print("DEBUG: cv2.imdecode returned None")
            return None
            
        if bgr.shape[0] < 50 or bgr.shape[1] < 50:  # Too small image
            print(f"DEBUG: Image too small: {bgr.shape}")
            return None
            
        print(f"DEBUG: Successfully decoded image: {bgr.shape}")
        return bgr
        
    except Exception as e:
        print(f"DEBUG: b64_to_bgr error: {str(e)}")
        return None


def extract_embedding(bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        model, det = load_insightface()
        if model is None or det is None:
            print("DEBUG: InsightFace model not loaded")
            return None
            
        faces = det.get(bgr)
        if not faces:
            return None
        # Choose largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding  # already L2-normalized 512-d
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        print(f"DEBUG: extract_embedding error: {e}")
        import traceback
        traceback.print_exc()
        return None

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
# Note: Rate limiting removed from face recognition endpoints to allow high-frequency check-in operations
@app.route("/api/recognize_open_gate", methods=["POST"])
def api_recognize_open_gate():
    if not CHECKIN_ENABLED:
        return jsonify({"ok": False, "error": "Check-in disabled by config"}), 400

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"ok": False, "error": "No JSON data provided"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid JSON data: {str(e)}"}), 400
    
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
            resp["member_id"] = best.gym_member_id
            # Create name from first_name + last_name
            first_name = best.first_name or ""
            last_name = best.last_name or ""
            if first_name and last_name:
                resp["name"] = f"{first_name} {last_name}"
            elif first_name:
                resp["name"] = first_name
            elif last_name:
                resp["name"] = last_name
            else:
                resp["name"] = best.email or f"Member {best.gym_member_id}"
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
                
                # DIOPTIMALKAN: Tambahkan informasi nama untuk frontend
                # Create name from first_name + last_name
                first_name = best.first_name or ""
                last_name = best.last_name or ""
                if first_name and last_name:
                    resp["name"] = f"{first_name} {last_name}"
                elif first_name:
                    resp["name"] = first_name
                elif last_name:
                    resp["name"] = last_name
                else:
                    resp["name"] = best.email or f"Member {best.gym_member_id}"
                resp["member_id"] = best.gym_member_id
                resp["success"] = True
                print(f"DEBUG: Response prepared with name: {resp['name']}")
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

# Endpoint untuk mengambil foto profil member
@app.route("/api/get_member_photo/<int:member_id>", methods=["GET"])
def api_get_member_photo(member_id):
    """Get member profile photo by member_id"""
    try:
        print(f"DEBUG: Getting member photo for member_id: {member_id}")
        # Login dengan member_id untuk mendapatkan token
        token = gym_login_with_memberid(member_id)
        if not token:
            print(f"DEBUG: Failed to login with member_id: {member_id}")
            return jsonify({"ok": False, "error": "Failed to login with member ID"}), 401
        
        # Ambil profile data
        profile = gym_get_profile(token)
        if not profile:
            print(f"DEBUG: Failed to fetch profile for member_id: {member_id}")
            return jsonify({"ok": False, "error": "Failed to fetch profile"}), 404
        
        print(f"DEBUG: Profile data for member_id {member_id}: {profile}")
        
        # Ambil URL foto profil
        profile_photo_url = profile.get("memberphoto")
        if not profile_photo_url:
            print(f"DEBUG: No profile photo found for member_id: {member_id}")
            return jsonify({"ok": False, "error": "No profile photo found"}), 404
        
        print(f"DEBUG: Profile photo URL for member_id {member_id}: {profile_photo_url}")
        
        # Ambil nama lengkap
        full_name = profile.get("fullname", f"Member {member_id}")
        first_name = profile.get("firstname", "")
        last_name = profile.get("surname", "")
        
        print(f"DEBUG: Returning successful response for member_id {member_id}")
        print(f"DEBUG: Photo URL: {profile_photo_url}")
        print(f"DEBUG: Full name: {full_name}")
        
        return jsonify({
            "ok": True,
            "member_id": member_id,
            "full_name": full_name,
            "first_name": first_name,
            "last_name": last_name,
            "photo_url": profile_photo_url
        })
        
    except Exception as e:
        print(f"DEBUG: Error getting member photo: {e}")
        return jsonify({"ok": False, "error": f"Error: {str(e)}"}), 500



# DIOPTIMALKAN: Fast recognition endpoint untuk skala besar
# Note: No rate limiting to support high-frequency face recognition requests
@app.route("/api/recognize_fast", methods=["POST"])
def api_recognize_fast():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image_b64")
        door_token = data.get("door_token")
        
        print(f"DEBUG: Received request - door_token: {door_token[:30] if door_token else 'None'}..., image_b64: {'present' if image_b64 else 'missing'}")
        
        if not door_token:
            return jsonify({"ok": False, "error": "door_token is required"}), 400
        
        # Verify door token and extract doorid
        token_result = verify_door_token(door_token, request)
        if not token_result:
            return jsonify({"ok": False, "error": "Invalid door token"}), 401
        doorid, _ = token_result
        
        if not image_b64:
            return jsonify({"ok": False, "error": "image_b64 is required"}), 400
        
        bgr = b64_to_bgr(image_b64)
        if bgr is None:
            return jsonify({"ok": False, "error": "Failed to decode image"}), 400
        
        model, _ = load_insightface()
        if model is None:
            return jsonify({"ok": False, "error": "Model not ready"}), 503
        
        faces = model.get(bgr)
        print(f"DEBUG: Face detection found {len(faces)} faces")
        if not faces:
            print("DEBUG: No faces detected by model")
            return jsonify({"ok": True, "matched": False, "error": "No face detected"})
        
        # Filter faces by size to avoid false positives
        # Only consider faces that are reasonably sized (not too small)
        min_face_area = 500  # Reduced minimum face area in pixels
        valid_faces = []
        for face in faces:
            bbox = face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if face_area >= min_face_area:
                valid_faces.append(face)
        
        print(f"DEBUG: After filtering, {len(valid_faces)} valid faces remain")
        if not valid_faces:
            print("DEBUG: No valid faces after size filtering")
            return jsonify({"ok": True, "matched": False, "error": "No valid face detected (face too small)"})
        
        # choose largest valid face
        f = max(valid_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        q = f.normed_embedding.astype(np.float32)
        
        best_member, best_score, second_score = find_best_match(q)
        matched = (
            best_member is not None and
            best_score >= SIM_THRESHOLD_MATCH and
            (best_score - (second_score if second_score is not None else 0.0)) >= TOP2_MARGIN
        )
        
        resp: Dict[str, Any] = {
            "ok": True,
            "matched": bool(matched),
            "best_score": float(best_score),
            "second_best": float(second_score),
            "bbox": [int(v) for v in f.bbox.tolist()],
        }
        
        if matched and best_member:
            member_id = int(best_member.gym_member_id)
            # Create name from first_name + last_name
            first_name = best_member.first_name or ""
            last_name = best_member.last_name or ""
            if first_name and last_name:
                name = f"{first_name} {last_name}"
            elif first_name:
                name = first_name
            elif last_name:
                name = last_name
            else:
                name = best_member.email or f"Member {member_id}"
            resp.update({
                "name": name,
                "candidate": {
                    "member_pk": int(best_member.member_pk),
                    "gym_member_id": member_id,
                    "email": best_member.email,
                    "first_name": best_member.first_name,
                    "last_name": best_member.last_name,
                }
            })
            
            # Cooldown
            if is_user_throttled(member_id):
                remaining = get_user_cooldown_remaining(member_id)
                resp["gate"] = {"throttled": True, "cooldown_remaining": remaining, "popup_style": "WARNING"}
                resp["member_id"] = member_id
                resp["name"] = name
                return jsonify(resp)
            
            # Open gate if enabled
            if CHECKIN_ENABLED:
                token = gym_login_with_memberid(member_id)
                if token:
                    gate_result = gym_open_gate(token, int(doorid))
                    if gate_result:
                        mark_user_recognized(member_id)
                        resp["success"] = True
                        # Check gate response to determine popup style
                        gate_popup_style = "GRANTED"  # Default
                        if isinstance(gate_result, dict):
                            gate_popup_style = gate_result.get("popup_style", "GRANTED")
                        resp["gate"] = {"throttled": False, "popup_style": gate_popup_style, "response": gate_result}
                        resp["member_id"] = member_id  # Add member_id to response
                        return jsonify(resp)
            
            # Recognized but gate not opened (fallback)
            mark_user_recognized(member_id)
            resp["success"] = True
            resp["gate"] = {"throttled": False, "popup_style": "GRANTED"}
            resp["member_id"] = member_id  # Add member_id to response
            return jsonify(resp)
        
        # Not matched
        resp["popup_style"] = "DENIED"
        return jsonify(resp)
    except Exception as e:
        print("DEBUG: api/recognize_fast error", e)
        return jsonify({"ok": False, "error": str(e)}), 500





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
    # Get user and branch info before clearing session
    user_id = session.get('user_id')
    branch_id = session.get('branch_id')
    
    # Remove user from branch tracking
    if user_id and branch_id:
        remove_user_from_branch(branch_id, user_id)
    
    # Clear session data
    session.pop('gym_token', None)
    session.pop('profile_data', None)
    session.pop('user_id', None)
    session.pop('branch_id', None)
    
    return jsonify({
        "ok": True,
        "message": "Logged out successfully"
    })





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


# ==================== WEBSOCKET HANDLERS ====================
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    client_info = {
        'id': client_id,
        'connected_at': datetime.now().isoformat(),
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown')
    }
    connected_devices[client_id] = client_info
    print(f"📱 Device connected: {client_id} from {request.remote_addr}")
    print(f"   Total connected devices: {len(connected_devices)}")
    
    # Send welcome message to client
    emit('connection_status', {
        'status': 'connected',
        'client_id': client_id,
        'message': 'Successfully connected to server'
    })
    
    # Broadcast device count to all clients
    socketio.emit('device_count', {'count': len(connected_devices)})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in connected_devices:
        device_info = connected_devices[client_id]
        del connected_devices[client_id]
        print(f"📴 Device disconnected: {client_id}")
        print(f"   Total connected devices: {len(connected_devices)}")
        
        # Broadcast device count to remaining clients
        socketio.emit('device_count', {'count': len(connected_devices)})


@socketio.on('heartbeat')
def handle_heartbeat(data):
    """Handle heartbeat from client to keep connection alive"""
    client_id = request.sid
    if client_id in connected_devices:
        connected_devices[client_id]['last_heartbeat'] = datetime.now().isoformat()
        emit('heartbeat_ack', {'status': 'alive', 'timestamp': datetime.now().isoformat()})


@socketio.on('register_device')
def handle_register_device(data):
    """Handle device registration with additional info"""
    client_id = request.sid
    if client_id in connected_devices:
        connected_devices[client_id].update({
            'device_name': data.get('device_name', 'Unknown'),
            'device_type': data.get('device_type', 'Unknown'),
            'location': data.get('location', 'Unknown'),
            'doorid': data.get('doorid', None)
        })
        print(f"📝 Device registered: {data.get('device_name')} at {data.get('location')}")
        emit('registration_success', {'status': 'registered', 'client_id': client_id})


# ==================== ADMIN CONTROL ENDPOINTS ====================
@app.route("/admin/control")
def admin_control_panel():
    """Admin panel for remote device control"""
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin_login'))
    
    return render_template_string(ADMIN_CONTROL_HTML)


@app.route("/api/admin/devices", methods=["GET"])
def api_get_connected_devices():
    """Get list of connected devices"""
    if not session.get('admin_authenticated'):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    
    devices_list = []
    for device_id, device_info in connected_devices.items():
        devices_list.append({
            'id': device_id,
            'name': device_info.get('device_name', 'Unknown'),
            'type': device_info.get('device_type', 'Unknown'),
            'location': device_info.get('location', 'Unknown'),
            'doorid': device_info.get('doorid', None),
            'ip': device_info.get('ip', 'Unknown'),
            'connected_at': device_info.get('connected_at', 'Unknown'),
            'last_heartbeat': device_info.get('last_heartbeat', 'N/A')
        })
    
    return jsonify({
        "ok": True,
        "devices": devices_list,
        "total": len(devices_list)
    })


@app.route("/api/admin/broadcast_refresh", methods=["POST"])
def api_broadcast_refresh():
    """Broadcast refresh command to all connected devices"""
    if not session.get('admin_authenticated'):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    
    try:
        data = request.get_json(force=True) or {}
        message = data.get('message', 'Server requested refresh')
        
        # Broadcast refresh command to all connected devices
        socketio.emit('force_refresh', {
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"🔄 Broadcast refresh to {len(connected_devices)} devices")
        
        return jsonify({
            "ok": True,
            "message": f"Refresh command sent to {len(connected_devices)} devices",
            "device_count": len(connected_devices)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/admin/refresh_device", methods=["POST"])
def api_refresh_specific_device():
    """Refresh a specific device by client_id"""
    if not session.get('admin_authenticated'):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    
    try:
        data = request.get_json(force=True)
        client_id = data.get('client_id')
        message = data.get('message', 'Server requested refresh')
        
        if not client_id:
            return jsonify({"ok": False, "error": "client_id is required"}), 400
        
        if client_id not in connected_devices:
            return jsonify({"ok": False, "error": "Device not connected"}), 404
        
        # Send refresh command to specific device
        socketio.emit('force_refresh', {
            'message': message,
            'timestamp': datetime.now().isoformat()
        }, room=client_id)
        
        print(f"🔄 Refresh command sent to device: {client_id}")
        
        return jsonify({
            "ok": True,
            "message": f"Refresh command sent to device {client_id}"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/admin/send_notification", methods=["POST"])
def api_send_notification():
    """Send notification to all devices"""
    if not session.get('admin_authenticated'):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    
    try:
        data = request.get_json(force=True)
        notification = data.get('notification', 'System notification')
        notification_type = data.get('type', 'info')  # info, warning, success, error
        
        socketio.emit('notification', {
            'message': notification,
            'type': notification_type,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            "ok": True,
            "message": f"Notification sent to {len(connected_devices)} devices"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------- Main --------------------
if __name__ == "__main__":
    startup_optimization()  # Now all functions are defined
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(app, host="0.0.0.0", port=APP_PORT, debug=True, allow_unsafe_werkzeug=True)
