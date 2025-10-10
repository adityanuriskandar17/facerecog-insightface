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

APP_PORT = int(os.getenv("PORT", 8081))
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
_RECOGNITION_COOLDOWN = 10  # 10 seconds cooldown per user

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


# Ini untuk login menuju retake START
LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FTL Face Gate - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }
        
        body { 
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; 
            margin: 0; 
            padding: 10px; 
            background: white;
            min-height: auto;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .login-container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
        }
        
        /* Mobile responsive styles for login */
        @media (max-width: 768px) {
            body {
                padding: 5px;
                margin: 0;
            }
            .login-container {
                padding: 25px;
                margin: 5px;
                border-radius: 8px;
                max-width: 100%;
                width: calc(100% - 10px);
            }
            .login-header h1 {
                font-size: 22px;
            }
            .login-header p {
                font-size: 13px;
            }
            .logo {
                width: 70px;
                height: 70px;
            }
        }
        
        @media (max-width: 480px) {
            body {
                padding: 3px;
            }
            .login-container {
                padding: 20px;
                margin: 3px;
                border-radius: 6px;
                width: calc(100% - 6px);
            }
            .login-header h1 {
                font-size: 20px;
            }
            .login-header p {
                font-size: 12px;
            }
            .logo {
                width: 60px;
                height: 60px;
            }
            .form-group input {
                padding: 12px;
                font-size: 16px;
                min-height: 48px;
            }
            .login-btn {
                padding: 12px;
                font-size: 16px;
                min-height: 48px;
            }
        }
        
        /* Extra small mobile devices */
        @media (max-width: 360px) {
            body {
                padding: 2px;
            }
            .login-container {
                padding: 15px;
                margin: 2px;
                border-radius: 4px;
                width: calc(100% - 4px);
            }
            .login-header h1 {
                font-size: 18px;
            }
            .login-header p {
                font-size: 11px;
            }
            .logo {
                width: 50px;
                height: 50px;
            }
            .form-group input {
                padding: 10px;
                font-size: 15px;
                min-height: 44px;
            }
            .login-btn {
                padding: 10px;
                font-size: 15px;
                min-height: 44px;
            }
        }
        
        /* Very small mobile devices (like iPhone SE) */
        @media (max-width: 320px) {
            .login-container {
                padding: 12px;
                margin: 1px;
                width: calc(100% - 2px);
            }
            .login-header h1 {
                font-size: 16px;
            }
            .login-header p {
                font-size: 10px;
            }
            .logo {
                width: 45px;
                height: 45px;
            }
            .form-group input {
                padding: 8px;
                font-size: 14px;
                min-height: 40px;
            }
            .login-btn {
                padding: 8px;
                font-size: 14px;
                min-height: 40px;
            }
        }
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .logo {
            width: 80px;
            height: 80px;
            background: #333;
            border-radius: 12px;
            margin: 0 auto 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-image: url('https://is1-ssl.mzstatic.com/image/thumb/Purple221/v4/59/86/27/598627f5-2f2a-a6a4-52b6-937cbff0ada5/AppIcon-0-0-1x_U007emarketing-0-7-0-0-85-220.png/1200x630wa.png');
            background-size: cover;
            background-position: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .login-header h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 24px;
            font-weight: 600;
        }
        .login-header p {
            color: #666;
            margin: 0;
            font-size: 14px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #333;
            font-weight: 500;
            font-size: 14px;
        }
        .form-group input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
            min-height: 48px;
            -webkit-appearance: none;
            appearance: none;
        }
        .form-group input:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
        }
        .password-input-container {
            position: relative;
            display: flex;
            align-items: center;
        }
        .password-input-container input {
            width: 100%;
            padding: 12px 45px 12px 12px; /* Added right padding for the icon */
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
            min-height: 48px;
            -webkit-appearance: none;
            appearance: none;
        }
        .password-toggle-btn {
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            background: none;
            border: none;
            cursor: pointer;
            color: #666;
            font-size: 16px;
            padding: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: color 0.2s;
        }
        .password-toggle-btn:hover {
            color: #007bff;
        }
        .password-toggle-btn:focus {
            outline: none;
            color: #007bff;
        }
        
        /* Mobile responsive form styles */
        @media (max-width: 768px) {
            .form-group {
                margin-bottom: 15px;
            }
            .form-group input {
                padding: 12px;
                font-size: 16px;
                min-height: 48px;
            }
            .password-input-container input {
                padding: 12px 40px 12px 12px;
                font-size: 16px;
                min-height: 48px;
            }
        }
        
        @media (max-width: 480px) {
            .form-group {
                margin-bottom: 12px;
            }
            .form-group input {
                padding: 12px;
                font-size: 16px;
                min-height: 48px;
            }
            .password-input-container input {
                padding: 12px 35px 12px 12px;
                font-size: 16px;
                min-height: 48px;
            }
        }
        
        /* Extra small mobile form styles */
        @media (max-width: 360px) {
            .form-group {
                margin-bottom: 10px;
            }
            .form-group input {
                padding: 10px;
                font-size: 15px;
                min-height: 44px;
            }
            .password-input-container input {
                padding: 10px 30px 10px 10px;
                font-size: 15px;
                min-height: 44px;
            }
        }
        
        @media (max-width: 320px) {
            .form-group {
                margin-bottom: 8px;
            }
            .form-group input {
                padding: 8px;
                font-size: 14px;
                min-height: 40px;
            }
            .password-input-container input {
                padding: 8px 25px 8px 8px;
                font-size: 14px;
                min-height: 40px;
            }
        }
        .login-btn {
            width: 100%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            min-height: 48px;
            touch-action: manipulation;
        }
        .login-btn:hover {
            background: #0056b3;
        }
        .login-btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        /* Mobile responsive button styles */
        @media (max-width: 768px) {
            .login-btn {
                padding: 10px;
                font-size: 14px;
            }
        }
        
        @media (max-width: 480px) {
            .login-btn {
                padding: 8px;
                font-size: 13px;
            }
        }
        
        /* Extra small mobile button styles */
        @media (max-width: 360px) {
            .login-btn {
                padding: 6px;
                font-size: 12px;
            }
        }
        
        @media (max-width: 320px) {
            .login-btn {
                padding: 5px;
                font-size: 11px;
            }
        }
        
        .back-btn {
            display: block;
            width: 100%;
            margin-top: 20px;
            color: white;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            padding: 12px;
            border: none;
            border-radius: 6px;
            background: #343a40;
            transition: all 0.2s;
            box-shadow: 0 2px 4px rgba(52, 58, 64, 0.2);
            text-align: center;
            box-sizing: border-box;
        }
        .back-btn:hover {
            background: #23272b;
            box-shadow: 0 4px 8px rgba(52, 58, 64, 0.3);
            transform: translateY(-1px);
        }
        .error-message {
            color: #dc3545;
            margin-top: 15px;
            text-align: center;
            padding: 10px;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 6px;
        }
        .success-message {
            color: #28a745;
            margin-top: 15px;
            text-align: center;
            padding: 10px;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 6px;
        }
        
        /* Mobile responsive message styles */
        @media (max-width: 768px) {
            .error-message, .success-message {
                margin-top: 12px;
                padding: 8px;
                font-size: 14px;
            }
        }
        
        @media (max-width: 480px) {
            .error-message, .success-message {
                margin-top: 10px;
                padding: 6px;
                font-size: 13px;
            }
        }
        
        /* Extra small mobile message styles */
        @media (max-width: 360px) {
            .error-message, .success-message {
                margin-top: 8px;
                padding: 5px;
                font-size: 12px;
            }
        }
        
        @media (max-width: 320px) {
            .error-message, .success-message {
                margin-top: 6px;
                padding: 4px;
                font-size: 11px;
            }
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <div class="logo"></div>
            <h1>FTL Face Registration</h1>
            <p>Login to Register Face</p>
        </div>
        
        <form id="loginForm">
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" id="email" name="email" placeholder="your.email@example.com" required>
            </div>
            
            <div class="form-group">
                <label for="password">Password</label>
                <div class="password-input-container">
                <input type="password" id="password" name="password" required>
                    <button type="button" id="togglePassword" class="password-toggle-btn">
                        <i class="fas fa-eye" id="passwordIcon"></i>
                    </button>
                </div>
            </div>
            
            <button type="submit" class="login-btn" id="loginBtn">
                <span>→</span>
                <span>Login</span>
            </button>
        </form>
        
        <div id="message"></div>
    </div>

    <script>
        // Password visibility toggle
        document.getElementById('togglePassword').addEventListener('click', function() {
            const passwordInput = document.getElementById('password');
            const passwordIcon = document.getElementById('passwordIcon');
            
            if (passwordInput.type === 'password') {
                passwordInput.type = 'text';
                passwordIcon.classList.remove('fa-eye');
                passwordIcon.classList.add('fa-eye-slash');
            } else {
                passwordInput.type = 'password';
                passwordIcon.classList.remove('fa-eye-slash');
                passwordIcon.classList.add('fa-eye');
            }
        });

        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const loginBtn = document.getElementById('loginBtn');
            const messageDiv = document.getElementById('message');
            
            loginBtn.disabled = true;
            loginBtn.innerHTML = '<span>→</span><span>Logging in...</span>';
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
                
                // Handle different response types
                let result;
                try {
                    result = await response.json();
                } catch (jsonError) {
                    // If JSON parsing fails, check if it's a rate limit error
                    if (response.status === 429) {
                        messageDiv.innerHTML = '<div class="error-message">⏱️ Too many login attempts. Please wait 60 seconds before trying again.</div>';
                        return;
                    } else {
                        messageDiv.innerHTML = '<div class="error-message">❌ Server error. Please try again later.</div>';
                        return;
                    }
                }
                
                if (result.ok) {
                    messageDiv.innerHTML = '<div class="success-message">✅ Login successful! Redirecting...</div>';
                    setTimeout(() => {
                        window.location.href = '/retake';
                    }, 1500);
                } else {
                    // Handle different error types
                    let errorMessage = result.error || 'Invalid credentials';
                    
                    // Add visual indicators for different error types
                    if (errorMessage.includes('Rate limit exceeded')) {
                        errorMessage = '⏱️ ' + errorMessage;
                    } else if (errorMessage.includes('Too many failed login attempts')) {
                        errorMessage = '🔒 ' + errorMessage;
                    } else if (errorMessage.includes('Branch is at maximum capacity')) {
                        errorMessage = '🏢 ' + errorMessage;
                    } else {
                        errorMessage = '❌ ' + errorMessage;
                    }
                    
                    messageDiv.innerHTML = '<div class="error-message">' + errorMessage + '</div>';
                }
            } catch (error) {
                messageDiv.innerHTML = '<div class="error-message">Login error: ' + error.message + '</div>';
            } finally {
                loginBtn.disabled = false;
                loginBtn.innerHTML = '<span>→</span><span>Login</span>';
            }
        });
    </script>
</body>
</html>
"""
# Ini untuk login menuju retake END



# Setelah Login pasti masuk sini START
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
      min-height: auto;
      padding: 20px;
    }
    
    .header {
      text-align: center;
      margin-bottom: 40px;
    }
    
    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      color: #0072bc;
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
      align-items: stretch;  /* DIOPTIMALKAN: Semua card memiliki tinggi yang sama */
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
      display: flex;
      flex-direction: column;  /* DIOPTIMALKAN: Konsisten layout untuk semua card */
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
      color: #0072bc;
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
      color: #0072bc;
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
      background: linear-gradient(135deg, #4ca7e5 0%, #0072bc 100%);
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
      max-width: 400px;
      height: 300px;  /* DIOPTIMALKAN: Proporsi yang lebih seimbang (4:3 aspect ratio) */
      background: linear-gradient(135deg, #4ca7e5 0%, #0072bc 100%);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: white;
      margin: 0 auto 20px;  /* DIOPTIMALKAN: Center alignment */
      position: relative;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .camera-container video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 12px;
      position: relative;
      z-index: 1;
    }
    
    /* DIOPTIMALKAN: Camera container alignment untuk card Validasi Member */
    .card .camera-container {
      flex: 1;  /* Mengisi ruang yang tersedia di card */
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    
    /* DIOPTIMALKAN: Responsive camera container untuk mobile */
    @media (max-width: 768px) {
      .camera-container {
        width: 100%;
        max-width: 350px;
        height: 250px;  /* Proporsi yang lebih baik untuk mobile */
        margin: 0 auto 15px;
      }
    }
    
    @media (max-width: 480px) {
      .camera-container {
        width: 100%;
        max-width: 300px;
        height: 200px;  /* Proporsi yang lebih baik untuk mobile kecil */
        margin: 0 auto 10px;
      }
    }
    
    .camera-placeholder {
      text-align: center;
      z-index: 2;
      position: relative;
    }
    
    .camera-placeholder i {
      font-size: 4rem;
      margin-bottom: 15px;
      opacity: 0.8;
      color: white;
    }
    
    .camera-placeholder p {
      font-size: 1.1rem;
      font-weight: 500;
      color: white;
    }
    
    /* Overlay untuk kamera */
    .camera-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 10;
      background: transparent;
      border-radius: 12px;
    }
    
    .btn-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    
    /* Sequential button styles for validation */
    .validation-sequential {
      transition: all 0.3s ease;
    }
    
    .validation-sequential:disabled {
      opacity: 0.5 !important;
      cursor: not-allowed !important;
      background-color: #6c757d !important;
    }
    
    .validation-sequential:not(:disabled) {
      opacity: 1 !important;
      cursor: pointer !important;
    }
    
    /* Sequential button styles for register */
    .register-sequential {
      transition: all 0.3s ease;
    }
    
    .register-sequential:disabled {
      opacity: 0.5 !important;
      cursor: not-allowed !important;
      background-color: #6c757d !important;
    }
    
    .register-sequential:not(:disabled) {
      opacity: 1 !important;
      cursor: pointer !important;
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
      border-color: #0072bc;
    }
    
    .btn-capture {
      background: white;
      color: #495057;
      border: 2px solid #e9ecef;
    }
    
    .btn-capture:hover {
      background: #f8f9fa;
      border-color: #0072bc;
    }
    
    .btn-capture:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    
    .btn-capture.loading {
      opacity: 0.7;
      cursor: not-allowed;
      position: relative;
    }
    
    .btn-capture.loading .fa-spinner {
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
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
      background: linear-gradient(135deg, #4ca7e5 0%, #0072bc 100%);
      color: white;
      border: none;
    }
    
    .btn-register:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    canvas { display: none; }
    
    .hidden { display: none; }
    
    /* Modern Stepper Styles */
    .stepper-container {
      margin: 30px 0;
      padding: 20px;
      background: linear-gradient(135deg, #f8fbff 0%, #e8f4fd 100%);
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(76, 167, 229, 0.15);
      border: 1px solid rgba(76, 167, 229, 0.2);
    }
    
    .stepper-horizontal {
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      padding: 20px 0;
      gap: 0;
    }
    
    .step-horizontal {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      position: relative;
      z-index: 2;
      margin: 0 20px;
    }
    
    .step-horizontal:not(:last-child)::after {
      content: '';
      position: absolute;
      top: 25px;
      left: 50px;
      width: calc(100% + 40px);
      height: 3px;
      background: #e9ecef;
      z-index: 1;
      border-radius: 2px;
    }
    
    .step-horizontal.completed:not(:last-child)::after {
      background: linear-gradient(135deg, #4CAF50, #388E3C);
      box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3);
    }
    
    .step-horizontal:hover .step-circle {
      transform: scale(1.1);
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }
    
    .step-horizontal.active .step-circle {
      background: linear-gradient(135deg, #4ca7e5 0%, #0072bc 100%);
      border-color: #0072bc;
      color: white;
      box-shadow: 0 4px 12px rgba(76, 167, 229, 0.4);
    }
    
    .step-horizontal.completed .step-circle {
      background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
      border-color: #28a745;
      color: white;
      box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
    }
    
    .step-circle {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      font-size: 18px;
      color: #6c757d;
      background: #f8f9fa;
      border: 3px solid #e9ecef;
      transition: all 0.3s ease;
      position: relative;
      flex-shrink: 0;
      z-index: 3;
      box-shadow: 0 0 0 2px white;
    }
    
    .step.active .step-circle {
      background: linear-gradient(135deg, #4ca7e5 0%, #0072bc 100%);
      border-color: #0072bc;
      color: white;
      box-shadow: 0 4px 12px rgba(76, 167, 229, 0.4);
    }
    
    .step.completed .step-circle {
      background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
      border-color: #28a745;
      color: white;
      box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
    }
    
    .step-number {
      display: block;
    }
    
    .step-check {
      display: none;
      font-size: 20px;
    }
    
    .step.completed .step-number {
      display: none;
    }
    
    .step.completed .step-check {
      display: block;
    }
    
    .step-content {
      text-align: center;
      max-width: 120px;
    }
    
    .step-title {
      font-size: 14px;
      font-weight: 600;
      color: #333;
      margin: 0;
      line-height: 1.2;
    }
    
    .step-horizontal.active .step-title {
      color: #0072bc;
      font-weight: 700;
    }
    
    .step-horizontal.completed .step-title {
      color: #28a745;
    }
    
    .step-description {
      font-size: 12px;
      color: #6c757d;
      margin: 0;
      line-height: 1.3;
    }
    
    .step-horizontal.active .step-description {
      color: #1976D2;
    }
    
    .step-horizontal.completed .step-description {
      color: #388E3C;
    }
    
    .step-action-btn {
      padding: 12px 24px;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.3s ease;
      min-width: 120px;
      justify-content: center;
    }
    
    .step-action-btn:not(:disabled) {
      background: linear-gradient(135deg, #4ca7e5 0%, #0072bc 100%);
      color: white;
      box-shadow: 0 4px 12px rgba(76, 167, 229, 0.3);
    }
    
    .step-action-btn:not(:disabled):hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(76, 167, 229, 0.4);
    }
    
    .step-action-btn:disabled {
      background: #e9ecef;
      color: #6c757d;
      cursor: not-allowed;
      opacity: 0.6;
    }
    
    .step-action-btn.reset-btn {
      background: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
      color: #212529;
    }
    
    .step-action-btn.reset-btn:hover:not(:disabled) {
      box-shadow: 0 6px 16px rgba(255, 193, 7, 0.4);
    }
    
    .stepper-progress {
      position: absolute;
      left: 45px;
      top: 70px;
      bottom: 70px;
      width: 4px;
      background: #e9ecef;
      border-radius: 2px;
      z-index: 1;
    }
    
    .progress-line {
      height: 0%;
      background: linear-gradient(180deg, #4ca7e5 0%, #0072bc 100%);
      border-radius: 2px;
      transition: height 0.5s ease;
      box-shadow: 0 2px 8px rgba(76, 167, 229, 0.3);
    }
    
    /* Loading Animation for Recognizing State - Retake Page */
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
      .stepper-container {
        margin: 20px 0;
        padding: 15px;
      }
      
      .step {
        padding: 15px;
        gap: 15px;
      }
      
      .step-circle {
        width: 40px;
        height: 40px;
        font-size: 16px;
      }
      
      .step-title {
        font-size: 16px;
      }
      
      .step-description {
        font-size: 13px;
      }
      
      .step-action-btn {
        padding: 10px 20px;
        font-size: 13px;
        min-width: 100px;
      }
      
      .stepper-progress {
        left: 35px;
      }
    }
    
    @media (max-width: 480px) {
      .step {
        flex-direction: column;
        text-align: center;
        gap: 10px;
      }
      
      .step-content {
        order: 2;
      }
      
      .step-action-btn {
        order: 3;
        width: 100%;
      }
      
      .stepper-progress {
        display: none;
      }
    }
    
    /* Pulse animation for completed steps */
    @keyframes pulse {
      0% {
        box-shadow: 0 0 20px rgba(0,123,255,0.6), 0 0 40px rgba(0,123,255,0.4);
      }
      50% {
        box-shadow: 0 0 30px rgba(0,123,255,0.8), 0 0 60px rgba(0,123,255,0.6);
      }
      100% {
        box-shadow: 0 0 20px rgba(0,123,255,0.6), 0 0 40px rgba(0,123,255,0.4);
      }
    }
    
    /* Responsive roadmap */
    @media (max-width: 768px) {
      .roadmap-container {
        padding: 0 10px !important;
        min-height: 70px !important;
      }
      
      .progress-line-bg {
        left: 10px !important;
        right: 10px !important;
        height: 2px !important;
      }
      
      .progress-line {
        left: 10px !important;
        height: 2px !important;
      }
      
      .step-circle {
        width: 35px !important;
        height: 35px !important;
        border-width: 2px !important;
        font-size: 14px !important;
      }
      
      .step-label {
        font-size: 10px !important;
        margin-top: 6px !important;
      }
    }
    
    @media (max-width: 480px) {
      .roadmap-container {
        padding: 0 5px !important;
        min-height: 60px !important;
      }
      
      .progress-line-bg {
        left: 5px !important;
        right: 5px !important;
        height: 1px !important;
      }
      
      .progress-line {
        left: 5px !important;
        height: 1px !important;
      }
      
      .step-circle {
        width: 30px !important;
        height: 30px !important;
        border-width: 1px !important;
        font-size: 12px !important;
      }
      
      .step-label {
        font-size: 9px !important;
        margin-top: 4px !important;
      }
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
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      width: 100%;
      box-sizing: border-box;
      position: relative;
      left: 0;
      right: 0;
      display: block;
      clear: both;
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
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
  <h1>Profile & Face Registration</h1>
      <p>Secure biometric authentication and identity verification system By FTL IT Developer</p>
    </div>
    
    <!-- Progress Roadmap -->
    <div class="card-stepper">
      <h3 style="color: #333; margin: 0 0 24px 0; text-align: center; font-size: 18px; font-weight: 600;">Validation Steps</h3>
      <div class="roadmap-container" style="display: flex; justify-content: space-between; align-items: center; position: relative; padding: 0 40px; min-height: 80px; width: 100%;">
        <!-- Progress Line -->
        <div class="progress-line-bg" style="position: absolute; top: 25px; left: 40px; right: 40px; height: 3px; background: #4ca7e5; z-index: 1; border-radius: 2px; opacity: 0.3;"></div>
        <div id="progressLine" class="progress-line" style="position: absolute; top: 25px; left: 40px; height: 3px; background: linear-gradient(90deg, #4ca7e5 0%, #0072bc 50%, #0037cf 100%); z-index: 2; transition: width 0.5s ease; width: 0%; border-radius: 2px;"></div>
        
        <!-- Step 1: Validasi -->
        <div class="roadmap-step" data-step="1" style="display: flex; flex-direction: column; align-items: center; z-index: 3; position: relative; flex: 1; min-width: 0;">
          <div class="step-circle" style="width: clamp(40px, 8vw, 60px); height: clamp(40px, 8vw, 60px); border-radius: 50%; background: #f8fbff; border: 3px solid #4ca7e5; display: flex; align-items: center; justify-content: center; color: #0072bc; font-weight: bold; font-size: clamp(16px, 3vw, 24px); transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(76, 167, 229, 0.2);">
            <i class="fas fa-check" style="display: none; font-size: clamp(14px, 2.5vw, 20px);"></i>
            <span class="step-number">1</span>
          </div>
          <div class="step-label" style="color: #0072bc; margin-top: clamp(8px, 2vw, 16px); font-size: clamp(11px, 2vw, 14px); text-align: center; font-weight: 500; line-height: 1.2; word-break: break-word;">Validasi</div>
        </div>
        
        <!-- Step 2: Face Recognition -->
        <div class="roadmap-step" data-step="2" style="display: flex; flex-direction: column; align-items: center; z-index: 3; position: relative; flex: 1; min-width: 0;">
          <div class="step-circle" style="width: clamp(40px, 8vw, 60px); height: clamp(40px, 8vw, 60px); border-radius: 50%; background: #f8fbff; border: 3px solid #4ca7e5; display: flex; align-items: center; justify-content: center; color: #0072bc; font-weight: bold; font-size: clamp(16px, 3vw, 24px); transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(76, 167, 229, 0.2);">
            <i class="fas fa-check" style="display: none; font-size: clamp(14px, 2.5vw, 20px);"></i>
            <span class="step-number">2</span>
          </div>
          <div class="step-label" style="color: #0072bc; margin-top: clamp(8px, 2vw, 16px); font-size: clamp(11px, 2vw, 14px); text-align: center; font-weight: 500; line-height: 1.2; word-break: break-word;">Face Recognition</div>
        </div>
        
        <!-- Step 3: Upload Foto -->
        <div class="roadmap-step" data-step="3" style="display: flex; flex-direction: column; align-items: center; z-index: 3; position: relative; flex: 1; min-width: 0;">
          <div class="step-circle" style="width: clamp(40px, 8vw, 60px); height: clamp(40px, 8vw, 60px); border-radius: 50%; background: #f8fbff; border: 3px solid #4ca7e5; display: flex; align-items: center; justify-content: center; color: #0072bc; font-weight: bold; font-size: clamp(16px, 3vw, 24px); transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(76, 167, 229, 0.2);">
            <i class="fas fa-check" style="display: none; font-size: clamp(14px, 2.5vw, 20px);"></i>
            <span class="step-number">3</span>
          </div>
          <div class="step-label" style="color: #0072bc; margin-top: clamp(8px, 2vw, 16px); font-size: clamp(11px, 2vw, 14px); text-align: center; font-weight: 500; line-height: 1.2; word-break: break-word;">Upload Foto</div>
        </div>
      </div>
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
          <img id="profile" alt="profile" style="display: none; transform: scaleX(-1);" />
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
      <h3>Validasi Member</h3>
        </div>
        <div class="camera-container" id="cameraContainer">
          <video id="video" autoplay playsinline muted style="display: none; transform: scaleX(-1);"></video>
            <canvas id="overlay" class="camera-overlay" style="display: none;"></canvas>
            <img id="capturedImage" alt="captured" style="display: none; width: 100%; height: 100%; object-fit: contain; border-radius: 12px;" />
            
            <!-- Loading Overlay for Recognizing State -->
            <div id="loadingOverlay" class="loading-overlay">
              <div class="loading-scan-line"></div>
              <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text-large">Scanning...</div>
                <div class="loading-text-small">Please wait while we scan your face</div>
              </div>
            </div>
          <div class="camera-placeholder" id="cameraPlaceholder">
            <i class="fas fa-camera"></i>
          </div>
        </div>
        <div class="btn-group">
          <button id="btnStart" class="btn btn-start validation-sequential" data-step="1">
            <b>1.</b>
            Start Camera
          </button>
          <button id="btnSnap" class="btn btn-capture validation-sequential" disabled data-step="2">
            <b>2.</b>
            Validasi
          </button>
          <button id="btnRegister" class="btn btn-register validation-sequential" disabled data-step="3">
            <b>3.</b>
            Register Face Recognition
          </button>
        </div>
        
        <!-- Camera Switch Buttons -->
        <div class="camera-switch-group" style="display: none; margin-top: 15px; gap: 10px; justify-content: center;">
          <button id="btnSwitchToFront" class="btn btn-switch" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 14px;">
            <i class="fas fa-user"></i>
            <span>Kamera Depan</span>
          </button>
          <button id="btnSwitchToBack" class="btn btn-switch" style="padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 14px;">
            <i class="fas fa-camera"></i>
            <span>Kamera Belakang</span>
          </button>
        </div>
        <canvas id="canvas" style="display: none;"></canvas>
        <pre id="out" style="display: none;"></pre>
      </div>
    </div>
  </div>
  
  <!-- Modal for Face Registration -->
  <div id="registerModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; padding: 24px; border-radius: 12px; max-width: 800px; width: 90%; max-height: 90vh; overflow-y: auto; position: relative;">
      <!-- Close Button di Pojok Kanan Atas -->
      <button id="btnCloseRegister" style="position: absolute; top: 16px; right: 16px; padding: 8px 16px; background: #dc3545; color: white; border: none; border-radius: 20px; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 14px; z-index: 10; box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);">
        <i class="fas fa-times"></i>
        <span>Close</span>
      </button>
      
      <h2 style="text-align: center; margin-bottom: 24px; color: #333;">Register Face Recognition</h2>
      
      <!-- Camera Section -->
      <div style="text-align: center; margin: 20px 0;">
        <div style="position: relative; width: 100%; max-width: 400px; height: 600px; margin: 0 auto; background: #111; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
          <video id="registerVideo" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); display: none; border-radius: 12px; position: absolute; top: 0; left: 0; z-index: 2; background: #000;"></video>
          <img id="registerCapturedImage" alt="captured" style="display: none; width: 100%; height: 100%; object-fit: contain; border-radius: 12px; position: absolute; top: 0; left: 0; z-index: 3; transform: scaleX(-1);" />
          <div id="cameraPlaceholder" style="width: 100%; height: 100%; background: #f8f9fa; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #6c757d; position: absolute; top: 0; left: 0; border-radius: 12px;">
            <i class="fas fa-camera" style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;"></i>
            <div style="font-size: 18px; font-weight: 500;">Camera Inactive</div>
          </div>
        </div>
      </div>
      
      <style>
        @media (max-width: 768px) {
          #registerVideo, #registerCapturedImage {
            object-fit: contain !important;
          }
        }
        @media (max-width: 480px) {
          #registerVideo, #registerCapturedImage {
            object-fit: contain !important;
          }
        }
      </style>
      
      <!-- Control Buttons -->
      <!-- Modern Horizontal Stepper Component -->
      <div class="stepper-container">
        <div class="stepper-horizontal">
          <!-- Step 1: Start Camera -->
          <div class="step-horizontal" data-step="1">
            <div class="step-circle" id="stepCircle1">
              <span class="step-number">1</span>
              <i class="fas fa-check step-check" style="display: none;"></i>
            </div>
            <div class="step-content">
              <div class="step-title">Start Camera</div>
              <div class="step-description">Initialize camera</div>
            </div>
          </div>
          
          <!-- Step 2: Capture Photo -->
          <div class="step-horizontal" data-step="2">
            <div class="step-circle" id="stepCircle2">
              <span class="step-number">2</span>
              <i class="fas fa-check step-check" style="display: none;"></i>
            </div>
            <div class="step-content">
              <div class="step-title">Capture Photo</div>
              <div class="step-description">Take final photo</div>
            </div>
          </div>
          
          <!-- Step 3: Update Photo -->
          <div class="step-horizontal" data-step="3">
            <div class="step-circle" id="stepCircle3">
              <span class="step-number">3</span>
              <i class="fas fa-check step-check" style="display: none;"></i>
            </div>
            <div class="step-content">
              <div class="step-title">Update Photo</div>
              <div class="step-description">Upload to GymMaster</div>
            </div>
          </div>
          
          <!-- Step 4: Burst Capture -->
          <div class="step-horizontal" data-step="4">
            <div class="step-circle" id="stepCircle4">
              <span class="step-number">4</span>
              <i class="fas fa-check step-check" style="display: none;"></i>
            </div>
            <div class="step-content">
              <div class="step-title">Burst Capture</div>
              <div class="step-description">Capture frames</div>
            </div>
          </div>
          
        </div>
      </div>
      
      <!-- Control Buttons -->
      <div style="display: flex; gap: 12px; justify-content: center; margin: 20px 0; flex-wrap: wrap;">
        <button id="btnStartRegister" class="step-action-btn" style="display: none; background: linear-gradient(135deg, #4ca7e5, #0072bc); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; box-shadow: 0 4px 12px rgba(76, 167, 229, 0.3); transition: all 0.3s ease;">
          <i class="fas fa-camera" style="margin-right: 8px;"></i>
          Start Camera
        </button>
        <button id="btnCapturePhoto" class="step-action-btn" style="background: linear-gradient(135deg, #28a745, #20c997); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3); transition: all 0.3s ease; opacity: 0.5; cursor: not-allowed;" disabled>
          <i class="fas fa-camera" style="margin-right: 8px;"></i>
          Capture Photo
        </button>
        <button id="btnUpdatePhoto" class="step-action-btn" style="background: linear-gradient(135deg, #6f42c1, #5a32a3); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; box-shadow: 0 4px 12px rgba(111, 66, 193, 0.3); transition: all 0.3s ease; opacity: 0.5; cursor: not-allowed;" disabled>
          <i class="fas fa-upload" style="margin-right: 8px;"></i>
          Update Photo
        </button>
        <button id="btnBurstCapture" class="step-action-btn" style="background: linear-gradient(135deg, #ff6b35, #f7931e); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3); transition: all 0.3s ease; opacity: 0.5; cursor: not-allowed;" disabled>
          <i class="fas fa-bolt" style="margin-right: 8px;"></i>
          Burst Capture
        </button>
        <button id="btnResetPhoto" class="step-action-btn reset-btn" style="background: linear-gradient(135deg, #ffc107, #ff9800); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3); transition: all 0.3s ease; opacity: 0.5; cursor: not-allowed;" disabled>
          <i class="fas fa-redo" style="margin-right: 8px;"></i>
          Reset Photo
        </button>
      </div>
      
      <!-- Progress and Status -->
      <div id="registerProgress" style="margin: 16px 0; font-size: 14px; color: #666; text-align: center; min-height: 20px;"></div>
      <canvas id="registerCanvas" style="display: none;"></canvas>
    </div>
  </div>
  
  <script>
    const log = document.getElementById('log');
    const out = document.getElementById('out');
    const img = document.getElementById('profile');

    function setLog(x){ log.textContent = typeof x==='string'? x : JSON.stringify(x,null,2); }
    function setOut(x){ out.textContent = typeof x==='string'? x : JSON.stringify(x,null,2); }

    let token=null; let stream=null; let registerStream=null; let burstInterval=null;
    
    // Function to stop validation camera
    function stopValidationCamera() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      
      // Stop current stream if exists
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
      }
      
      const video = document.getElementById('video');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      const overlay = document.getElementById('overlay');
      
      if (video) {
        video.style.display = 'none';
        video.srcObject = null;
      }
      
      if (cameraPlaceholder) {
        cameraPlaceholder.style.display = 'flex';
      }
      
      if (overlay) {
        overlay.style.display = 'none';
      }
      
      // Hide camera switch buttons
      const switchGroup = document.querySelector('.camera-switch-group');
      if (switchGroup) {
        switchGroup.style.display = 'none';
      }
      
      // Reset validation buttons
      const btnStart = document.getElementById('btnStart');
      const btnSnap = document.getElementById('btnSnap');
      const btnCapturePhoto = document.getElementById('btnCapturePhoto');
      
      // Keep Start Camera button always enabled for restart functionality
      if (btnStart) btnStart.disabled = false;
      if (btnSnap) btnSnap.disabled = true;
      if (btnCapturePhoto) btnCapturePhoto.disabled = true;
    }

    // Sequential validation button state management
    let currentValidationStep = 1;
    const maxValidationSteps = 3;
    
    function updateValidationButtonStates() {
      console.log('updateValidationButtonStates called');
      const buttons = [
        'btnStart',      // Step 1
        'btnSnap',       // Step 2 (Validasi - bisa digunakan berulang kali)
        'btnRegister'    // Step 3
      ];
      
      buttons.forEach((btnId, index) => {
        const button = document.getElementById(btnId);
        console.log(`Processing button ${btnId}:`, !!button);
        if (button) {
          const stepNumber = index + 1;
          
          // Button Start Camera (Step 1) - always enabled for restart functionality
          if (btnId === 'btnStart') {
            console.log('Processing btnStart: Always enabled for restart functionality');
            button.disabled = false;
            button.style.opacity = '1';
            button.style.cursor = 'pointer';
            return;
          }
          
          // Button Validasi (Step 2) - skip sequential logic, handle separately
          if (btnId === 'btnSnap') {
            console.log('Processing btnSnap: Skipping sequential logic - handled separately');
            // Don't change button state here, it's handled directly in camera start
            return;
          }
          
          // Button Register (Step 3) - only enabled after successful validation
          if (btnId === 'btnRegister') {
            const validationCompleted = sessionStorage.getItem('validation_completed') === 'true';
            if (validationCompleted) {
              console.log('Processing btnRegister: Enabled after successful validation');
              button.disabled = false;
              button.style.opacity = '1';
              button.style.cursor = 'pointer';
            } else {
              console.log('Processing btnRegister: Disabled - validation not completed');
              button.disabled = true;
              button.style.opacity = '0.5';
              button.style.cursor = 'not-allowed';
            }
            return;
          }
          
          // Other buttons follow sequential logic
          if (stepNumber <= currentValidationStep) {
            button.disabled = false;
            button.style.opacity = '1';
            button.style.cursor = 'pointer';
          } else {
            button.disabled = true;
            button.style.opacity = '0.5';
            button.style.cursor = 'not-allowed';
          }
        }
      });
    }
    
    function nextValidationStep() {
      if (currentValidationStep < maxValidationSteps) {
        currentValidationStep++;
        updateValidationButtonStates();
        console.log(`Advanced to validation step ${currentValidationStep}`);
      }
    }
    
    function resetValidationSteps() {
      currentValidationStep = 1;
      sessionStorage.removeItem('validation_completed');
      updateValidationButtonStates();
      console.log('Reset to validation step 1');
    }
    
    // Sequential register button state management
    let currentRegisterStep = 1;
    const maxRegisterSteps = 4;
    
    function updateRegisterButtonStates() {
      // Manual control - only reset states, don't override manual activation
      // This function is now mainly for resetting states when modal opens
      console.log('updateRegisterButtonStates called - not overriding manual button states');
    }
    
    function nextRegisterStep() {
      if (currentRegisterStep < maxRegisterSteps) {
        currentRegisterStep++;
        updateRegisterButtonStates();
        updateStepperStates();
        console.log(`Advanced to register step ${currentRegisterStep}`);
      } else {
        // All steps completed, show completion message
        console.log('All register steps completed, showing completion message...');
        showFinalCountdown();
      }
    }
    
    // Auto close modal after photo update (for auto register flow)
    let isModalClosing = false;
    function autoCloseModalAfterUpdate() {
      if (isModalClosing) {
        console.log('Modal is already closing, skipping...');
        return;
      }
      isModalClosing = true;
      console.log('Photo update completed, showing completion message...');
      showFinalCountdown();
    }
    
    function resetRegisterSteps() {
      currentRegisterStep = 1;
      updateRegisterButtonStates();
      updateStepperStates();
      isModalClosing = false;
      isFinalCountdownShown = false;
      console.log('Reset to register step 1');
    }
    
    // Stepper Management Functions
    function updateStepperStates() {
      const steps = document.querySelectorAll('.step-horizontal');
      console.log(`Updating stepper states: currentRegisterStep = ${currentRegisterStep}, total steps = ${steps.length}`);
      
      steps.forEach((step, index) => {
        const stepNumber = index + 1;
        const stepCircle = step.querySelector('.step-circle');
        const stepNumberSpan = stepCircle.querySelector('.step-number');
        const stepCheck = stepCircle.querySelector('.step-check');
        
        console.log(`Processing step ${stepNumber}:`, {
          stepNumber,
          currentRegisterStep,
          isCompleted: stepNumber < currentRegisterStep,
          isActive: stepNumber === currentRegisterStep
        });
        
        // Remove all state classes
        step.classList.remove('active', 'completed');
        
        if (stepNumber < currentRegisterStep) {
          // Completed steps
          step.classList.add('completed');
          if (stepNumberSpan) stepNumberSpan.style.display = 'none';
          if (stepCheck) stepCheck.style.display = 'block';
          console.log(`Step ${stepNumber} marked as completed`);
        } else if (stepNumber === currentRegisterStep) {
          // Current active step
          step.classList.add('active');
          if (stepNumberSpan) stepNumberSpan.style.display = 'block';
          if (stepCheck) stepCheck.style.display = 'none';
          console.log(`Step ${stepNumber} marked as active`);
        } else {
          // Future steps
          if (stepNumberSpan) stepNumberSpan.style.display = 'block';
          if (stepCheck) stepCheck.style.display = 'none';
          console.log(`Step ${stepNumber} marked as future`);
        }
      });
      
      console.log(`Stepper updated: currentRegisterStep = ${currentRegisterStep}`);
    }
    
    function nextStepperStep() {
      if (currentRegisterStep < maxRegisterSteps) {
        currentRegisterStep++;
        updateRegisterButtonStates();
        updateStepperStates();
        console.log(`Advanced to stepper step ${currentRegisterStep}`);
      } else {
        // All steps completed, show completion message
        console.log('All stepper steps completed, showing completion message...');
        showFinalCountdown();
      }
    }
    
    function resetStepperSteps() {
      currentRegisterStep = 1;
      updateRegisterButtonStates();
      updateStepperStates();
      console.log('Reset to stepper step 1');
    }

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
          
          // Hide loading message and display profile info
          if (loadingProfile) {
            loadingProfile.style.display = 'none';
          }
          
          // Display profile info with styled fields
          profileInfo.innerHTML = `
            <div class="profile-field">
              <i class="fas fa-user"></i>
              <span>${profile.fullname || 'N/A'}</span>
            </div>
            <div class="profile-field">
              <i class="fas fa-envelope"></i>
              <span>${profile.email || 'N/A'}</span>
            </div>
          `;
          
          // Load profile photo
          const photoContainer = document.getElementById('photoContainer');
          const photoPlaceholder = document.getElementById('photoPlaceholder');
          const statusMessage = document.getElementById('statusMessage');
          
          if (profile.memberphoto) {
            setLog('Loading profile photo: ' + profile.memberphoto);
            
            // Add cache-busting parameter to ensure fresh photo is loaded
            const photoUrl = profile.memberphoto + (profile.memberphoto.includes('?') ? '&' : '?') + 't=' + Date.now();
            img.src = photoUrl;
            
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
            if (loadingProfile) {
              loadingProfile.style.display = 'none';
            }
            profileInfo.innerHTML = '<div style="color: red;">Not logged in. Please go back to login page.</div>';
          }
        }
      } catch (e) {
        if (loadingProfile) {
          loadingProfile.style.display = 'none';
        }
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
    
    // Progress Roadmap Functions for Profile Page
    function updateProgressRoadmap() {
      // Check if roadmap elements exist
      const roadmapContainer = document.querySelector('.roadmap-container');
      if (!roadmapContainer) {
        console.warn('Roadmap container not found');
        return;
      }
      
      let completedSteps = 0;
      
      // Check if validation is completed (Step 1: Validasi)
      const isValidationCompleted = sessionStorage.getItem('validation_completed') === 'true';
      if (isValidationCompleted) {
        completedSteps = 1;
        updateStepStatus(1, true);
      } else {
        updateStepStatus(1, false);
      }
      
      // Check if face recognition is registered (Step 2: Face Recognition)
      const hasFaceData = sessionStorage.getItem('face_registered') || false;
      if (hasFaceData) {
        completedSteps = 2;
        updateStepStatus(2, true);
      } else {
        updateStepStatus(2, false);
      }
      
      // Check if photo is uploaded to GymMaster (Step 3: Upload Foto)
      const hasPhotoUploaded = sessionStorage.getItem('photo_uploaded') || false;
      if (hasPhotoUploaded) {
        completedSteps = 3;
        updateStepStatus(3, true);
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
      if (!progressLine) return;
      
      const totalSteps = 3;
      const progressPercentage = (completedSteps / totalSteps) * 100;
      
      progressLine.style.width = progressPercentage + '%';
      progressLine.style.background = 'linear-gradient(90deg, #4ca7e5 0%, #0072bc 50%, #0037cf 100%)';
      progressLine.style.boxShadow = '0 4px 12px rgba(0, 55, 207, 0.4)';
    }
    
    // Initialize progress roadmap when page loads
    updateProgressRoadmap();
    
    // Initialize validation button states
    updateValidationButtonStates();
    
    // Initialize Register button based on validation status
    const btnRegister = document.getElementById('btnRegister');
    if (btnRegister) {
      const validationCompleted = sessionStorage.getItem('validation_completed') === 'true';
      if (validationCompleted) {
        btnRegister.disabled = false;
        btnRegister.style.opacity = '1';
        btnRegister.style.cursor = 'pointer';
        console.log('Button Register initialized as enabled - validation completed');
      } else {
        btnRegister.disabled = true;
        btnRegister.style.opacity = '0.5';
        btnRegister.style.cursor = 'not-allowed';
        console.log('Button Register initialized as disabled - validation not completed');
      }
    }

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
          
          // Clear validation status
          sessionStorage.removeItem('validation_completed');
          
          // Redirect to login page
          window.location.href = '/';
        } else {
          alert('Logout failed: ' + (j.error || 'Unknown error'));
        }
      } catch (e) {
        setLog('Logout error: ' + e.message);
        alert('Logout error: ' + e.message);
      }
    }

    // Camera switching variables
    let currentCameraFacing = 'user'; // 'user' for front, 'environment' for back
    let currentStream = null;
    
        // Add event handler for Start button on retake page
    document.getElementById('btnStart').onclick = async () => {
      try {
        setOut('Starting camera...');
        console.log('Start Camera button clicked - attempting to start camera');
        console.log('Current URL:', window.location.href);
        console.log('Protocol:', location.protocol);
        console.log('Hostname:', location.hostname);
        
        // Check if we're on HTTPS or localhost
        const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        if (!isSecure) {
          setOut('Camera requires HTTPS or localhost. Current protocol: ' + location.protocol);
          console.warn('Camera access requires HTTPS or localhost');
          return;
        }
        
        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          setOut('Camera not supported in this browser');
          console.error('getUserMedia not supported');
          return;
        }
        
        // Check available devices
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          console.log('Available devices:', devices);
          const videoDevices = devices.filter(device => device.kind === 'videoinput');
          console.log('Video devices found:', videoDevices.length);
        } catch (e) {
          console.log('Could not enumerate devices:', e);
        }
        
        await startCameraWithFacing('user'); // Start with front camera by default
        
        setOut('Camera started. Click "Validasi" to start face recognition.');
        
        // Show camera switch buttons
        const switchGroup = document.querySelector('.camera-switch-group');
        if (switchGroup) {
          switchGroup.style.display = 'flex';
        }
        
        // Directly enable button Validasi after camera start
        const btnSnap = document.getElementById('btnSnap');
        if (btnSnap) {
          btnSnap.disabled = false;
          btnSnap.style.opacity = '1';
          btnSnap.style.cursor = 'pointer';
          console.log('Button Validasi directly enabled after camera start');
        }
        
        // Keep Start Camera button enabled for restart functionality
        console.log('Camera started successfully - Start button remains enabled for restart');
      } catch (e) {
        setOut('Camera error: ' + e.message);
        console.error('Camera start error:', e);
        console.error('Error name:', e.name);
        console.error('Error message:', e.message);
        
        // Show specific error messages
        if (e.name === 'NotAllowedError') {
          setOut('Camera access denied. Please allow camera permission and try again.');
        } else if (e.name === 'NotFoundError') {
          setOut('No camera found. Please check if camera is connected.');
        } else if (e.name === 'NotSupportedError') {
          setOut('Camera not supported in this browser.');
        } else if (e.name === 'OverconstrainedError') {
          setOut('Camera constraints not supported. Trying with basic constraints...');
          // Try with basic constraints as fallback
          try {
            const basicConstraints = { video: true };
            console.log('Trying with basic constraints:', basicConstraints);
            const stream = await navigator.mediaDevices.getUserMedia(basicConstraints);
            const video = document.getElementById('video');
            video.srcObject = stream;
            video.style.display = 'block';
            document.getElementById('cameraPlaceholder').style.display = 'none';
            setOut('Camera started with basic constraints');
          } catch (fallbackError) {
            setOut('Camera failed even with basic constraints: ' + fallbackError.message);
          }
        } else {
          setOut('Camera error: ' + e.message);
        }
      }
    };
    
    // Camera switching functions
    async function startCameraWithFacing(facingMode) {
      try {
        // Stop existing stream if any
        if (currentStream) {
          console.log('Stopping existing camera stream...');
          currentStream.getTracks().forEach(track => track.stop());
          currentStream = null;
        }
        
        const video = document.getElementById('video');
        const cameraContainer = document.getElementById('cameraContainer');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        const overlay = document.getElementById('overlay');
        
        // Request camera with simplified constraints
        console.log(`Requesting camera with facing mode: ${facingMode}`);
        
        // Use simple constraints that work on most devices
        const constraints = {
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 640, max: 1280 },
            height: { ideal: 480, max: 720 },
            frameRate: { ideal: 15, max: 30 }
          }
        };
        
        console.log('Camera constraints:', constraints);
        
        try {
          currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (constraintError) {
          console.log('First attempt failed, trying with basic constraints:', constraintError);
          // Fallback to basic constraints
          const basicConstraints = { video: true };
          currentStream = await navigator.mediaDevices.getUserMedia(basicConstraints);
        }
        console.log('Camera stream obtained:', currentStream);
        
        video.srcObject = currentStream;
        video.style.display = 'block';
        cameraPlaceholder.style.display = 'none';
        
        // Wait for video to load
        video.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
        };
        
        // Setup overlay canvas
        if (overlay) {
          overlay.style.display = 'block';
          overlay.width = cameraContainer.offsetWidth;
          overlay.height = cameraContainer.offsetHeight;
        }
        
        currentCameraFacing = facingMode;
        updateCameraSwitchButtons();
        
        console.log(`Camera started successfully with facing mode: ${facingMode}`);
      } catch (e) {
        console.error('Camera error:', e);
        throw e;
      }
    }
    
    function updateCameraSwitchButtons() {
      const btnFront = document.getElementById('btnSwitchToFront');
      const btnBack = document.getElementById('btnSwitchToBack');
      
      if (currentCameraFacing === 'user') {
        // Currently using front camera
        if (btnFront) {
          btnFront.style.background = '#28a745';
          btnFront.style.opacity = '1';
        }
        if (btnBack) {
          btnBack.style.background = '#6c757d';
          btnBack.style.opacity = '0.7';
        }
      } else {
        // Currently using back camera
        if (btnFront) {
          btnFront.style.background = '#6c757d';
          btnFront.style.opacity = '0.7';
        }
        if (btnBack) {
          btnBack.style.background = '#28a745';
          btnBack.style.opacity = '1';
        }
      }
    }
    
    // Camera switch event handlers
    document.getElementById('btnSwitchToFront').onclick = async () => {
      if (currentCameraFacing === 'user') return; // Already using front camera
      
      try {
        setOut('Switching to front camera...');
        await startCameraWithFacing('user');
        setOut('Switched to front camera');
      } catch (e) {
        setOut('Failed to switch to front camera: ' + e.message);
      }
    };
    
    document.getElementById('btnSwitchToBack').onclick = async () => {
      if (currentCameraFacing === 'environment') return; // Already using back camera
      
      try {
        setOut('Switching to back camera...');
        await startCameraWithFacing('environment');
        setOut('Switched to back camera');
      } catch (e) {
        setOut('Failed to switch to back camera: ' + e.message);
      }
    };

    // Add event handler for Capture & Compare button
    document.getElementById('btnSnap').onclick = async () => {
      // Button Validasi bisa digunakan berulang kali, tidak mengikuti sequential logic
      
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas');
      const btnSnap = document.getElementById('btnSnap');
      
      if (!video.srcObject) {
        setOut('Please start camera first');
        return;
      }
      
      // Add loading state
      btnSnap.disabled = true;
      btnSnap.classList.add('loading');
      btnSnap.innerHTML = '<i class="fas fa-spinner"></i> Validasi...';
      
      try {
        setOut('Capturing photo...');
        
        // Create canvas with proper dimensions matching video
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        
        // Capture current frame with correct aspect ratio
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
        
        setOut('Comparing with profile photo...');
        
        // Refresh profile data before comparison to avoid session conflicts
        try {
          const profileRefresh = await fetch('/api/get_profile', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
          });
          const profileData = await profileRefresh.json();
          if (!profileData.ok) {
            console.log('Profile refresh failed, proceeding with comparison anyway');
          }
        } catch (e) {
          console.log('Profile refresh error, proceeding with comparison anyway:', e);
        }
        
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
            
            // Mark validation as completed and update roadmap
            sessionStorage.setItem('validation_completed', 'true');
            updateProgressRoadmap();
            
            // Enable Register button after successful validation
            currentValidationStep = 3;
            updateValidationButtonStates();
            
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
      } finally {
        // Reset button state
        btnSnap.disabled = false;
        btnSnap.classList.remove('loading');
        btnSnap.innerHTML = '<b>2. </b>Validasi';
        
        // Button Validasi tidak maju ke step berikutnya, bisa digunakan berulang kali
      }
    };

    // Add event handler for Capture Photo button
    document.getElementById('btnCapturePhoto').onclick = async () => {
      // Check if we're in register modal or main page
      const registerVideo = document.getElementById('registerVideo');
      const registerCapturedImage = document.getElementById('registerCapturedImage');
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas');
      const capturedImage = document.getElementById('capturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      if (registerVideo && registerCapturedImage) {
        // Handle register modal capture
        if (currentRegisterStep !== 2) {
          console.log('Button 2 can only be clicked when current step is 2');
          return;
        }
        
        if (!registerStream) {
          document.getElementById('registerProgress').textContent = 'Please start camera first';
          return;
        }
        
        try {
          // Create canvas with proper dimensions matching video
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = registerVideo.videoWidth;
          tempCanvas.height = registerVideo.videoHeight;
          
          const ctx = tempCanvas.getContext('2d');
          ctx.drawImage(registerVideo, 0, 0, tempCanvas.width, tempCanvas.height);
          const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
          
          // Show captured image
          registerCapturedImage.src = dataUrl;
          registerCapturedImage.style.display = 'block';
          registerVideo.style.display = 'none';
          
          // Ensure image loads properly
          registerCapturedImage.onload = () => {
            console.log('Captured image loaded successfully');
          };
          
          // Update button states
          document.getElementById('btnCapturePhoto').disabled = true;
          document.getElementById('btnBurstCapture').disabled = true;
          
          // Disable Capture Photo and Burst Capture buttons with proper styling
          const captureButton = document.getElementById('btnCapturePhoto');
          captureButton.style.opacity = '0.5';
          captureButton.style.cursor = 'not-allowed';
          
          const burstButton = document.getElementById('btnBurstCapture');
          burstButton.style.opacity = '0.5';
          burstButton.style.cursor = 'not-allowed';
          
          // Enable Reset Photo button with proper styling
          const resetButton = document.getElementById('btnResetPhoto');
          resetButton.disabled = false;
          resetButton.style.opacity = '1';
          resetButton.style.cursor = 'pointer';
          
          // Enable the update photo button with proper styling
          const updateButton = document.getElementById('btnUpdatePhoto');
          updateButton.disabled = false;
          updateButton.style.opacity = '1';
          updateButton.style.cursor = 'pointer';
          
          document.getElementById('registerProgress').textContent = 'Photo captured! Review the result and click "Update to GymMaster" if satisfied, or "Reset Photo" to retake.';
          
          // Move to step 3 (Update Photo) after photo capture completed
          currentRegisterStep = 3;
          updateStepperStates();
          
          console.log('After photo capture: Only Update Photo and Reset Photo buttons are enabled');
          
        } catch (e) {
          document.getElementById('registerProgress').textContent = 'Capture error: ' + e.message;
        }
      } else {
        // Handle main page capture
        if (!video.srcObject) {
          setOut('Please start camera first');
          return;
        }
        
        try {
          setOut('Capturing photo...');
          
          // Create canvas with proper dimensions matching video
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = video.videoWidth;
          tempCanvas.height = video.videoHeight;
          
          // Capture current frame with correct aspect ratio
          const ctx = tempCanvas.getContext('2d');
          ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
          const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
          
          // Show captured image
          capturedImage.src = dataUrl;
          capturedImage.style.display = 'block';
          video.style.display = 'none';
          cameraPlaceholder.style.display = 'none';
          
          // Update button states
          document.getElementById('btnCapturePhoto').disabled = true;
          document.getElementById('btnBurstCapture').disabled = true;
          
          // Disable Capture Photo and Burst Capture buttons with proper styling
          const captureButton = document.getElementById('btnCapturePhoto');
          captureButton.style.opacity = '0.5';
          captureButton.style.cursor = 'not-allowed';
          
          const burstButton = document.getElementById('btnBurstCapture');
          burstButton.style.opacity = '0.5';
          burstButton.style.cursor = 'not-allowed';
          
          // Enable Reset Photo button with proper styling
          const resetButton = document.getElementById('btnResetPhoto');
          resetButton.disabled = false;
          resetButton.style.opacity = '1';
          resetButton.style.cursor = 'pointer';
          
          // Enable the update photo button with proper styling
          const updateButton = document.getElementById('btnUpdatePhoto');
          updateButton.disabled = false;
          updateButton.style.opacity = '1';
          updateButton.style.cursor = 'pointer';
          
          setOut('Photo captured! Review the result and click "Update to GymMaster" if satisfied, or "Reset Photo" to retake.');
          
          // Move to step 3 (Update Photo) after photo capture completed
          currentRegisterStep = 3;
          updateStepperStates();
          
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
      }
    };

    // Add event handler for Reset Photo button
    document.getElementById('btnResetPhoto').onclick = () => {
      // Check if we're in register modal or main page
      const registerVideo = document.getElementById('registerVideo');
      const registerCapturedImage = document.getElementById('registerCapturedImage');
      const video = document.getElementById('video');
      const capturedImage = document.getElementById('capturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      if (registerVideo && registerCapturedImage) {
        // Reset register modal camera view
        registerVideo.style.display = 'block';
        registerCapturedImage.style.display = 'none';
        
        // Reset to step 2 (Capture Photo step) so user can capture again
        currentRegisterStep = 2;
        updateStepperStates();
        
        // Enable Capture Photo button with proper styling
        const captureButton = document.getElementById('btnCapturePhoto');
        captureButton.disabled = false;
        captureButton.style.opacity = '1';
        captureButton.style.cursor = 'pointer';
        
        // Disable Update Photo button since no photo is captured yet
        const updateButton = document.getElementById('btnUpdatePhoto');
        updateButton.disabled = true;
        updateButton.style.opacity = '0.5';
        updateButton.style.cursor = 'not-allowed';
        
        // Keep Reset Photo button enabled
        const resetButton = document.getElementById('btnResetPhoto');
        resetButton.disabled = false;
        resetButton.style.opacity = '1';
        resetButton.style.cursor = 'pointer';
        
        document.getElementById('registerProgress').textContent = 'Photo reset. You can now capture a new photo.';
        console.log('Reset to register step 3 - Capture Photo enabled');
      } else {
        // Reset main page camera view
        video.style.display = 'block';
        capturedImage.style.display = 'none';
        cameraPlaceholder.style.display = 'none';
        
        // Update button states - Reset button should always be enabled
        document.getElementById('btnCapturePhoto').disabled = false;
        document.getElementById('btnUpdatePhoto').disabled = true;
        document.getElementById('btnResetPhoto').disabled = false; // Keep reset button enabled
        
        setOut('Photo reset. You can now capture a new photo.');
      }
    };

    // Unified event handler for Update Profile Photo button (both main page and register modal)
    document.getElementById('btnUpdatePhoto').onclick = async () => {
      // Check if we're in register modal by looking for registerCapturedImage
      const registerCapturedImage = document.getElementById('registerCapturedImage');
      const capturedImage = document.getElementById('capturedImage');
      
      let targetImage, isRegisterModal = false;
      
      // Determine context and target image
      if (registerCapturedImage && registerCapturedImage.src && registerCapturedImage.style.display !== 'none') {
        // We're in register modal
        targetImage = registerCapturedImage;
        isRegisterModal = true;
        
        // Check if we're in the correct step for register modal
        console.log('Current register step:', currentRegisterStep);
        if (currentRegisterStep !== 3) {
          console.log('Button 3 can only be clicked when current step is 3, but current step is:', currentRegisterStep);
          return;
        }
      } else if (capturedImage && capturedImage.src && capturedImage.style.display !== 'none') {
        // We're in main page
        targetImage = capturedImage;
        isRegisterModal = false;
      } else {
        // No captured image available
        if (isRegisterModal) {
          document.getElementById('registerProgress').textContent = 'Please capture a photo first';
        } else {
          setOut('Please capture a photo first');
        }
        return;
      }
      
      try {
        // Use the captured image data
        const dataUrl = targetImage.src;
        
        // Show confirmation dialog only for main page
        if (!isRegisterModal) {
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
          
          if (!result.isConfirmed) {
            return;
          }
        }
        
        // Update progress message based on context
        if (isRegisterModal) {
          document.getElementById('registerProgress').textContent = 'Updating profile photo...';
        } else {
          setOut('Updating profile photo...');
        }
          
          // Send to update API
          const r = await fetch('/api/update_profile_photo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: dataUrl.split(',')[1] })
          });
          
          const j = await r.json();
          
          if (j.ok) {
            // Mark photo upload as completed
            sessionStorage.setItem('photo_uploaded', 'true');
            updateProgressRoadmap();
            
            if (isRegisterModal) {
              // Success for register modal
              document.getElementById('registerProgress').textContent = 'Profile photo updated successfully! You can now use Burst Capture.';
              document.getElementById('btnUpdatePhoto').disabled = true;
              
              // Move to step 4 (Burst Capture) after successful Update Photo
              console.log('Update Photo successful, moving to step 4');
              currentRegisterStep = 4;
              
              // Add small delay to ensure DOM is ready
              setTimeout(() => {
                updateStepperStates();
                console.log('Stepper updated to step 4, step 3 should now show completed');
                
                // Additional check to ensure step 3 is marked as completed
                const step3 = document.querySelector('[data-step="3"]');
                if (step3) {
                  console.log('Step 3 element found:', step3);
                  console.log('Step 3 classes:', step3.classList.toString());
                  const step3Circle = step3.querySelector('.step-circle');
                  const step3Check = step3Circle.querySelector('.step-check');
                  console.log('Step 3 check element:', step3Check);
                  console.log('Step 3 check display:', step3Check ? step3Check.style.display : 'not found');
                }
              }, 100);
              
              // Ensure video stream is still active for burst capture
              const registerVideo = document.getElementById('registerVideo');
              const registerCapturedImage = document.getElementById('registerCapturedImage');
              
              if (registerVideo && registerStream) {
                console.log('Video stream still active for burst capture');
                // Show video and hide captured image for burst capture
                registerVideo.style.display = 'block';
                registerVideo.style.opacity = '1';
                if (registerCapturedImage) {
                  registerCapturedImage.style.display = 'none';
                }
              } else {
                console.warn('Video stream not available for burst capture');
              }
              
              // Enable Burst Capture button after successful Update Photo
              const burstButton = document.getElementById('btnBurstCapture');
              burstButton.disabled = false;
              burstButton.style.opacity = '1';
              burstButton.style.cursor = 'pointer';
              
              // Success notification for register modal
              Swal.fire({
                title: '✅ Success!',
                text: 'Success updated photos. Burst Capture is now available!',
                icon: 'success',
                confirmButtonText: 'OK',
                confirmButtonColor: '#28a745',
                timer: 3000,
                timerProgressBar: true
              });
            } else {
              // Success for main page
              setOut('Profile photo updated successfully!');
              
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
            }
          } else {
            // Error handling based on context
            const errorMsg = 'Failed to update profile photo: ' + (j.error || 'Unknown error');
            if (isRegisterModal) {
              document.getElementById('registerProgress').textContent = errorMsg;
            } else {
              setOut(errorMsg);
              Swal.fire({
                title: '❌ Update Failed',
                text: errorMsg,
                icon: 'error',
                confirmButtonText: 'OK',
                confirmButtonColor: '#dc3545'
              });
            }
          }
      } catch (e) {
        // Error handling based on context
        const errorMsg = 'Update error: ' + e.message;
        if (isRegisterModal) {
          document.getElementById('registerProgress').textContent = errorMsg;
        } else {
          setOut(errorMsg);
          Swal.fire({
            title: '❌ Error',
            text: errorMsg,
            icon: 'error',
            confirmButtonText: 'OK',
            confirmButtonColor: '#dc3545'
          });
        }
      }
    };

    // Auto Register Process Functions
    let autoRegisterTimeout = null;
    
    function showNotificationPopup(message, duration = 5000) {
      let countdown = Math.ceil(duration / 1000);
      
      return Swal.fire({
        title: '⏳ Processing...',
        html: `
          <div style="text-align: center;">
            <div style="font-size: 18px; margin-bottom: 20px;">${message}</div>
            <div style="font-size: 48px; font-weight: bold; color: #2196F3; margin: 20px 0;">
              <span id="countdown-timer">${countdown}</span>
            </div>
            <div style="font-size: 14px; color: #666;">Memulai dalam...</div>
            <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; margin-top: 15px;">
              <div id="progress-bar" style="width: 0%; height: 8px; background: linear-gradient(90deg, #2196F3, #21CBF3); border-radius: 10px; transition: width 0.1s ease;"></div>
            </div>
          </div>
        `,
        showConfirmButton: false,
        allowOutsideClick: false,
        allowEscapeKey: false,
        didOpen: () => {
          const timerElement = document.getElementById('countdown-timer');
          const progressBar = document.getElementById('progress-bar');
          let timeLeft = duration;
          
          const interval = setInterval(() => {
            timeLeft -= 100;
            const secondsLeft = Math.ceil(timeLeft / 1000);
            
            if (timerElement) {
              timerElement.textContent = secondsLeft;
            }
            
            if (progressBar) {
              const progress = ((duration - timeLeft) / duration) * 100;
              progressBar.style.width = progress + '%';
            }
            
            if (timeLeft <= 0) {
              clearInterval(interval);
            }
          }, 100);
        }
      });
    }
    
    async function startAutoRegisterProcess() {
      console.log('Starting automatic register process...');
      
      try {
        // Step 1: Start Camera immediately (no countdown)
        console.log('Starting camera immediately...');
        await startRegisterCamera();
        
        // Update stepper to show step 1 completed and step 2 active
        currentRegisterStep = 2;
        updateStepperStates();
        
        // Enable Capture Photo button after camera starts
        const captureButton = document.getElementById('btnCapturePhoto');
        captureButton.disabled = false;
        captureButton.style.opacity = '1';
        captureButton.style.cursor = 'pointer';
        
        // Burst Capture button is now disabled until Update Photo is successful
        document.getElementById('registerProgress').textContent = 'Camera started! Click "Capture Photo" to take a photo first.';
        console.log('Camera started automatically, step 1 completed, step 2 active, Capture Photo enabled');
        
      } catch (error) {
        console.error('Register process error:', error);
        document.getElementById('registerProgress').textContent = 'Camera start failed: ' + error.message;
      }
    }
    
    async function startRegisterCamera() {
      return new Promise((resolve, reject) => {
        try {
          console.log('Starting register camera...');
          document.getElementById('registerProgress').textContent = 'Starting camera...';
          
          // Stop validation camera if running
          stopValidationCamera();
          
          // Stop any existing stream first
          if (registerStream) {
            registerStream.getTracks().forEach(track => track.stop());
            registerStream = null;
          }
          
          navigator.mediaDevices.getUserMedia({ 
            video: { 
              width: { ideal: 720 }, 
              height: { ideal: 1280 },
              facingMode: 'user'
            } 
          }).then(stream => {
            registerStream = stream;
            const video = document.getElementById('registerVideo');
            const placeholder = document.getElementById('cameraPlaceholder');
            
            if (video && placeholder) {
              video.srcObject = stream;
              video.style.display = 'block';
              placeholder.style.display = 'none';
              
              // Add event listeners for better handling
              video.onloadedmetadata = () => {
                console.log('Video metadata loaded');
                video.play().then(() => {
                  console.log('Video started playing');
                  document.getElementById('registerProgress').textContent = 'Camera started successfully!';
                  
                  // Enable Capture Photo button after camera starts
                  const captureButton = document.getElementById('btnCapturePhoto');
                  captureButton.disabled = false;
                  captureButton.style.opacity = '1';
                  captureButton.style.cursor = 'pointer';
                  
                  // Burst capture button remains disabled until Update Photo success
                  console.log('Camera ready, Capture Photo enabled, but Burst Capture disabled until Update Photo success');
                  
                  resolve();
                }).catch(err => {
                  console.error('Video play error:', err);
                  reject(err);
                });
              };
              
              video.onerror = (err) => {
                console.error('Video error:', err);
                reject(new Error('Video failed to load'));
              };
              
            } else {
              reject(new Error('Video elements not found'));
            }
          }).catch(err => {
            console.error('getUserMedia error:', err);
            document.getElementById('registerProgress').textContent = 'Camera access denied or not available';
            reject(err);
          });
          
        } catch (e) {
          console.error('Camera error:', e);
          document.getElementById('registerProgress').textContent = 'Camera error: ' + e.message;
          reject(e);
        }
      });
    }
    
    async function performBurstCapture() {
      return new Promise(async (resolve, reject) => {
        try {
          console.log('Performing burst capture...');
          
          // Step 1: Validate server connection and user profile first
          const progress = document.getElementById('registerProgress');
          progress.textContent = 'Validating server connection...';
          
          // Check if user is logged in and profile is available
          const profileResponse = await fetch('/api/get_profile', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
          });
          
          if (!profileResponse.ok) {
            throw new Error('User not logged in or profile not available');
          }
          
          const profileData = await profileResponse.json();
          if (!profileData.ok) {
            throw new Error('Failed to get user profile');
          }
          
          progress.textContent = 'Server validated. Starting burst capture...';
          
          if (!registerStream) {
            reject(new Error('Camera not started'));
            return;
          }
          
          const registerVideo = document.getElementById('registerVideo');
          
          if (!registerVideo) {
            reject(new Error('Video element not found'));
            return;
          }
          
          let capturedFrames = [];
          let countdown = 5;
          
          // Create prominent countdown display with enhanced styling
          const countdownDisplay = document.createElement('div');
          countdownDisplay.id = 'burstCountdown';
          countdownDisplay.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #ff6b35, #f7931e);
            color: white;
            padding: 30px 40px;
            border-radius: 20px;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 10px 30px rgba(255, 107, 53, 0.4);
            z-index: 10000;
            animation: pulse 1s infinite;
            border: 3px solid #fff;
          `;
          
          // Add pulse animation
          const style = document.createElement('style');
          style.textContent = `
            @keyframes pulse {
              0% { transform: translate(-50%, -50%) scale(1); }
              50% { transform: translate(-50%, -50%) scale(1.05); }
              100% { transform: translate(-50%, -50%) scale(1); }
            }
            @keyframes flash {
              0%, 100% { background: linear-gradient(135deg, #ff6b35, #f7931e); }
              50% { background: linear-gradient(135deg, #ff8c42, #ffa726); }
            }
          `;
          document.head.appendChild(style);
          
          countdownDisplay.innerHTML = `
            <div style="font-size: 18px; margin-bottom: 10px;">🚀 BURST CAPTURE STARTING</div>
            <div id="countdownNumber" style="font-size: 48px; font-weight: 900; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">${countdown}</div>
            <div style="font-size: 16px; margin-top: 10px;">Get ready to capture 25 frames!</div>
          `;
          
          document.body.appendChild(countdownDisplay);
          
          progress.textContent = `Burst capture starting in ${countdown} seconds...`;
          
          // Enhanced countdown with visual feedback
          const countdownInterval = setInterval(() => {
            countdown--;
            const countdownNumber = document.getElementById('countdownNumber');
            if (countdownNumber) {
              countdownNumber.textContent = countdown;
              // Flash effect for last 3 seconds
              if (countdown <= 3) {
                countdownDisplay.style.animation = 'flash 0.5s infinite';
              }
            }
            progress.textContent = `Burst capture starting in ${countdown} seconds...`;
            if (countdown <= 0) {
              clearInterval(countdownInterval);
              document.body.removeChild(countdownDisplay);
              startBurstCapture();
            }
          }, 1000);
          
          function startBurstCapture() {
            // Create prominent frame capture notification
            const frameNotification = document.createElement('div');
            frameNotification.id = 'frameCaptureNotification';
            frameNotification.style.cssText = `
              position: fixed;
              top: 20px;
              left: 50%;
              transform: translateX(-50%);
              background: linear-gradient(135deg, #4CAF50, #45a049);
              color: white;
              padding: 20px 30px;
              border-radius: 15px;
              font-size: 20px;
              font-weight: bold;
              text-align: center;
              box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
              z-index: 10000;
              animation: slideDown 0.5s ease-out;
              border: 2px solid #fff;
              min-width: 300px;
            `;
            
            // Add slide animation
            const frameStyle = document.createElement('style');
            frameStyle.textContent = `
              @keyframes slideDown {
                from { transform: translateX(-50%) translateY(-100px); opacity: 0; }
                to { transform: translateX(-50%) translateY(0); opacity: 1; }
              }
              @keyframes slideUp {
                from { transform: translateX(-50%) translateY(0); opacity: 1; }
                to { transform: translateX(-50%) translateY(-100px); opacity: 0; }
              }
              @keyframes framePulse {
                0% { transform: translateX(-50%) scale(1); }
                50% { transform: translateX(-50%) scale(1.02); }
                100% { transform: translateX(-50%) scale(1); }
              }
            `;
            document.head.appendChild(frameStyle);
            
            frameNotification.innerHTML = `
              <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                <div style="font-size: 24px;">📸</div>
                <div>
                  <div style="font-size: 16px; margin-bottom: 5px;">CAPTURING FRAMES</div>
                  <div id="frameCount" style="font-size: 28px; font-weight: 900;">0 / 25</div>
                </div>
                <div style="font-size: 24px;">📸</div>
              </div>
              <div style="margin-top: 10px; font-size: 14px; opacity: 0.9;">Please stay still and look at the camera</div>
            `;
            
            document.body.appendChild(frameNotification);
            
            progress.textContent = 'Capturing photos... (5 seconds)';
            
            let burstInterval = setInterval(() => {
              // Create canvas with proper dimensions matching video
              const tempCanvas = document.createElement('canvas');
              tempCanvas.width = registerVideo.videoWidth;
              tempCanvas.height = registerVideo.videoHeight;
              
              const ctx = tempCanvas.getContext('2d');
              ctx.drawImage(registerVideo, 0, 0, tempCanvas.width, tempCanvas.height);
              const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
              capturedFrames.push(dataUrl);
              
              // Update frame count with animation
              const frameCount = document.getElementById('frameCount');
              if (frameCount) {
                frameCount.textContent = `${capturedFrames.length} / 25`;
                frameNotification.style.animation = 'framePulse 0.3s ease-out';
          setTimeout(() => {
                  frameNotification.style.animation = '';
                }, 300);
              }
              
              progress.textContent = `Photo ${capturedFrames.length} captured...`;
            }, 200); // Take photo every 200ms
            
            setTimeout(async () => {
              clearInterval(burstInterval);
              burstInterval = null;
              
              // Remove frame notification
              const frameNotification = document.getElementById('frameCaptureNotification');
              if (frameNotification) {
                document.body.removeChild(frameNotification);
              }
              
              // Show completion notification
              const completionNotification = document.createElement('div');
              completionNotification.style.cssText = `
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: linear-gradient(135deg, #2196F3, #1976D2);
                color: white;
                padding: 20px 30px;
                border-radius: 15px;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
                box-shadow: 0 8px 25px rgba(33, 150, 243, 0.4);
                z-index: 10000;
                animation: slideDown 0.5s ease-out;
                border: 2px solid #fff;
                min-width: 300px;
              `;
              completionNotification.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 15px;">
                  <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="processing-animation">
                      <div class="processing-dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                      </div>
                    </div>
                    <div style="color: #fff; font-size: 16px; font-weight: 500;">Processing ${capturedFrames.length} frames...</div>
                  </div>
                </div>
                <style>
                  .processing-animation {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                  }
                  .processing-dots {
                    display: flex;
                    gap: 4px;
                    align-items: center;
                  }
                  .dot {
                    width: 8px;
                    height: 8px;
                    background: #fff;
                    border-radius: 50%;
                    animation: processingBounce 1.4s ease-in-out infinite both;
                  }
                  .dot:nth-child(1) { animation-delay: -0.32s; }
                  .dot:nth-child(2) { animation-delay: -0.16s; }
                  .dot:nth-child(3) { animation-delay: 0s; }
                  @keyframes processingBounce {
                    0%, 80%, 100% {
                      transform: scale(0.8);
                      opacity: 0.5;
                    }
                    40% {
                      transform: scale(1.2);
                      opacity: 1;
                    }
                  }
                </style>
              `;
              document.body.appendChild(completionNotification);
              
              progress.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px; justify-content: center;">
                  <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <div>Sending photos for face encoding...</div>
                </div>
                <style>
                  .loading-dots {
                    display: inline-flex;
                    gap: 4px;
                  }
                  .loading-dots span {
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                    background-color: #2196F3;
                    animation: loadingDots 1.4s infinite ease-in-out both;
                  }
                  .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
                  .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
                  .loading-dots span:nth-child(3) { animation-delay: 0s; }
                  @keyframes loadingDots {
                    0%, 80%, 100% { 
                      transform: scale(0);
                      opacity: 0.5;
                    } 
                    40% { 
                      transform: scale(1);
                      opacity: 1;
                    }
                  }
                </style>
              `;
              
              // Send frames to server for encoding
              const r = await fetch('/api/register_face', {
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: capturedFrames })
              });
              const j = await r.json();
              
              // Remove completion notification
              if (completionNotification) {
                document.body.removeChild(completionNotification);
              }
              
              // Clear any existing notifications
              const existingSuccess = document.querySelector('[id*="success"]');
              if (existingSuccess) {
                document.body.removeChild(existingSuccess);
              }
              
              if (j.ok) {
                // Show success popup first
                const successNotification = document.createElement('div');
                successNotification.id = 'faceEncodingSuccess';
                successNotification.style.cssText = `
                  position: fixed;
                  top: 50%;
                  left: 50%;
                  transform: translate(-50%, -50%);
                  background: linear-gradient(135deg, #4CAF50, #45a049);
                  color: white;
                  padding: 30px 40px;
                  border-radius: 20px;
                  font-size: 24px;
                  font-weight: bold;
                  text-align: center;
                  box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);
                  z-index: 10000;
                  animation: successPulse 0.5s ease-out;
                  border: 3px solid #fff;
                  min-width: 350px;
                `;
                
                // Add success animation
                const successStyle = document.createElement('style');
                successStyle.textContent = `
                  @keyframes successPulse {
                    0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
                    50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
                    100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
                  }
                `;
                document.head.appendChild(successStyle);
                
                successNotification.innerHTML = `
                  <div style="font-size: 48px; margin-bottom: 15px;">✅</div>
                  <div style="font-size: 20px; margin-bottom: 10px;">SUCCESS!</div>
                  <div style="font-size: 16px;">Face encoding saved to database</div>
                `;
                
                document.body.appendChild(successNotification);
                
                // Remove success popup after 2 seconds and show countdown for photo capture
                setTimeout(() => {
                  if (document.body.contains(successNotification)) {
                    document.body.removeChild(successNotification);
                  }
                  
                  // Update stepper to show burst capture completed
                  updateStepperStates();
                  
                  // Show countdown popup for photo capture
                  showCountdownForPhotoCapture();
          }, 2000);
                
                console.log('Face registration successful:', j);
                
                // Enable the capture photo button instead of auto-advancing
                document.getElementById('btnCapturePhoto').disabled = false;
                
                resolve();
              } else {
                throw new Error(j.error || 'Failed to register face');
              }
            }, 5000); // Capture for 5 seconds
          }
          
        } catch (e) {
          console.error('Burst capture error:', e);
          document.getElementById('registerProgress').textContent = 'Burst capture error: ' + e.message;
          reject(e);
        }
      });
    }
    
    function showCountdownForPhotoCapture() {
      // Remove automatic countdown - just enable the capture button
      const progress = document.getElementById('registerProgress');
      progress.textContent = 'Burst capture completed! Click "Capture Photo" to take your final photo.';
      
      // Enable the capture photo button with proper styling
      const captureButton = document.getElementById('btnCapturePhoto');
      captureButton.disabled = false;
      captureButton.style.opacity = '1';
      captureButton.style.cursor = 'pointer';
      console.log('Capture Photo button enabled in showCountdownForPhotoCapture');
    }
    
    let isFinalCountdownShown = false;
    
    function updateStepperToCompleted() {
      const steps = document.querySelectorAll('.step-horizontal');
      
      steps.forEach((step, index) => {
        const stepNumber = index + 1;
        const stepCircle = step.querySelector('.step-circle');
        const stepNumberSpan = stepCircle.querySelector('.step-number');
        const stepCheck = stepCircle.querySelector('.step-check');
        
        // Mark all steps as completed
        step.classList.remove('active');
        step.classList.add('completed');
        
        // Show checkmark and hide number
        if (stepNumberSpan) stepNumberSpan.style.display = 'none';
        if (stepCheck) stepCheck.style.display = 'block';
        
        // Update circle styling to completed state
        stepCircle.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
        stepCircle.style.borderColor = '#28a745';
        stepCircle.style.color = 'white';
        stepCircle.style.boxShadow = '0 4px 12px rgba(40, 167, 69, 0.4)';
        
        // Update text colors
        const stepTitle = step.querySelector('.step-title');
        const stepDescription = step.querySelector('.step-description');
        if (stepTitle) {
          stepTitle.style.color = '#28a745';
          stepTitle.style.fontWeight = '700';
        }
        if (stepDescription) {
          stepDescription.style.color = '#388E3C';
        }
      });
      
      console.log('Stepper updated to show all steps completed');
    }
    
    function showFinalCountdown() {
      // Remove automatic countdown - just show completion message
      const progress = document.getElementById('registerProgress');
      progress.textContent = 'Registration completed successfully! You can now close the modal.';
      
      // Update stepper to show all steps completed with checkmarks
      updateStepperToCompleted();
      
      return Promise.resolve();
    }
    
    async function captureRegisterPhoto() {
      return new Promise((resolve, reject) => {
        try {
          const registerVideo = document.getElementById('registerVideo');
          const registerCapturedImage = document.getElementById('registerCapturedImage');
          
          if (!registerVideo || !registerCapturedImage) {
            reject(new Error('Required elements not found'));
            return;
          }
          
          // Create canvas with proper dimensions matching video
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = registerVideo.videoWidth;
          tempCanvas.height = registerVideo.videoHeight;
          
          const ctx = tempCanvas.getContext('2d');
          ctx.drawImage(registerVideo, 0, 0, tempCanvas.width, tempCanvas.height);
          const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
          
          // Show captured image
          registerCapturedImage.src = dataUrl;
          registerCapturedImage.style.display = 'block';
          registerVideo.style.display = 'none';
          
          // Update button states
          document.getElementById('btnCapturePhoto').disabled = true;
          document.getElementById('btnBurstCapture').disabled = true;
          
          // Disable Capture Photo and Burst Capture buttons with proper styling
          const captureButton = document.getElementById('btnCapturePhoto');
          captureButton.style.opacity = '0.5';
          captureButton.style.cursor = 'not-allowed';
          
          const burstButton = document.getElementById('btnBurstCapture');
          burstButton.style.opacity = '0.5';
          burstButton.style.cursor = 'not-allowed';
          
          // Enable Reset Photo button with proper styling
          const resetButton = document.getElementById('btnResetPhoto');
          resetButton.disabled = false;
          resetButton.style.opacity = '1';
          resetButton.style.cursor = 'pointer';
          
          // Enable the update photo button with proper styling
          const updateButton = document.getElementById('btnUpdatePhoto');
          updateButton.disabled = false;
          updateButton.style.opacity = '1';
          updateButton.style.cursor = 'pointer';
          
          document.getElementById('registerProgress').textContent = 'Photo captured successfully!';
          
          // Move to step 3 (Update Photo) after photo capture completed
          currentRegisterStep = 3;
          updateStepperStates();
          
          console.log('After photo capture: Only Update Photo and Reset Photo buttons are enabled');
          resolve();
          
        } catch (e) {
          console.error('Capture error:', e);
          document.getElementById('registerProgress').textContent = 'Capture error: ' + e.message;
          reject(e);
        }
      });
    }
    
    async function updateRegisterPhoto() {
      return new Promise(async (resolve, reject) => {
        try {
          const registerCapturedImage = document.getElementById('registerCapturedImage');
          
          if (!registerCapturedImage || !registerCapturedImage.src) {
            reject(new Error('No captured image found'));
            return;
          }
          
          // Convert image to base64 for JSON API
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          const img = new Image();
          
          img.onload = async () => {
            try {
              canvas.width = img.width;
              canvas.height = img.height;
              ctx.drawImage(img, 0, 0);
              const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
              
              // Send as JSON with base64 data
          const r = await fetch('/api/update_profile_photo', {
            method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_b64: dataUrl.split(',')[1] })
          });
          
          const j = await r.json();
          
          if (j.ok) {
            // Mark photo upload as completed
            sessionStorage.setItem('photo_uploaded', 'true');
            updateProgressRoadmap();
            
            // Show success popup
            const successNotification = document.createElement('div');
            successNotification.id = 'photoUpdateSuccess';
            successNotification.style.cssText = `
              position: fixed;
              top: 50%;
              left: 50%;
              transform: translate(-50%, -50%);
              background: linear-gradient(135deg, #28a745, #20c997);
              color: white;
              padding: 30px 40px;
              border-radius: 15px;
              box-shadow: 0 10px 30px rgba(40, 167, 69, 0.3);
              z-index: 10000;
              text-align: center;
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
              animation: slideDown 0.5s ease-out;
            `;
            successNotification.innerHTML = `
              <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                <div style="font-size: 32px;">✅</div>
                <div>
                  <div style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Photo Updated Successfully!</div>
                  <div style="font-size: 16px; opacity: 0.9;">Your profile photo has been updated to GymMaster</div>
                </div>
                <div style="font-size: 32px;">✅</div>
              </div>
            `;
            
            document.body.appendChild(successNotification);
            
            // Auto remove success popup after 3 seconds
            setTimeout(() => {
              if (successNotification && document.body.contains(successNotification)) {
                successNotification.style.animation = 'slideUp 0.5s ease-out';
                setTimeout(() => {
                  if (document.body.contains(successNotification)) {
                    document.body.removeChild(successNotification);
                  }
                }, 500);
              }
            }, 3000);
            
            document.getElementById('registerProgress').textContent = 'Profile photo updated successfully!';
            document.getElementById('btnUpdatePhoto').disabled = true;
            
            // Success notification for register modal
            Swal.fire({
              title: '✅ Success!',
              text: 'Success updated photos',
              icon: 'success',
              confirmButtonText: 'OK',
              confirmButtonColor: '#28a745',
              timer: 3000,
              timerProgressBar: true
            });
            
            resolve();
          } else {
            reject(new Error(j.message || 'Upload failed'));
          }
            } catch (e) {
              reject(e);
            }
          };
          
          img.onerror = () => {
            reject(new Error('Failed to load image'));
          };
          
          img.src = registerCapturedImage.src;
          
        } catch (e) {
          console.error('Update error:', e);
          document.getElementById('registerProgress').textContent = 'Update error: ' + e.message;
          // Reject to trigger error popup in the calling function (after capture photos)
          reject(e);
        }
      });
    }
    
    function closeRegisterModal() {
      console.log('Closing register modal...');
      
      // Stop register stream
      if (registerStream) {
        registerStream.getTracks().forEach(track => track.stop());
        registerStream = null;
      }
      
      // Reset modal elements
      const registerVideo = document.getElementById('registerVideo');
      const registerCapturedImage = document.getElementById('registerCapturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      if (registerVideo) {
        registerVideo.style.display = 'none';
        registerVideo.srcObject = null;
      }
      if (registerCapturedImage) {
        registerCapturedImage.style.display = 'none';
        registerCapturedImage.src = '';
      }
      if (cameraPlaceholder) {
        cameraPlaceholder.style.display = 'flex';
      }
      
      // Reset button states
      document.getElementById('btnCapturePhoto').disabled = true;
      document.getElementById('btnUpdatePhoto').disabled = true;
      document.getElementById('btnResetPhoto').disabled = true;
      document.getElementById('btnBurstCapture').disabled = true;
      
      document.getElementById('registerProgress').textContent = '';
      
      // Clean up any remaining popups
      const finalCountdown = document.getElementById('finalCountdown');
      if (finalCountdown && document.body.contains(finalCountdown)) {
        document.body.removeChild(finalCountdown);
        console.log('Removed final countdown popup');
      }
      
      // Clean up other possible popups
      const burstCountdown = document.getElementById('burstCountdown');
      if (burstCountdown && document.body.contains(burstCountdown)) {
        document.body.removeChild(burstCountdown);
        console.log('Removed burst countdown popup');
      }
      
      const frameNotification = document.getElementById('frameCaptureNotification');
      if (frameNotification && document.body.contains(frameNotification)) {
        document.body.removeChild(frameNotification);
        console.log('Removed frame notification popup');
      }
      
      // Force close modal
      const registerModal = document.getElementById('registerModal');
      if (registerModal) {
        console.log('Closing modal element:', registerModal);
        registerModal.style.display = 'none';
        registerModal.style.visibility = 'hidden';
        registerModal.style.opacity = '0';
        console.log('Modal closed successfully');
      } else {
        console.error('Modal element not found!');
      }
      
      // Reset register steps when closing modal
      resetRegisterSteps();
    }

    // Face Registration Modal Controls
    document.getElementById('btnRegister').onclick = () => {
      // Button Register can be clicked directly without validation
      
      console.log('Opening register modal...');
      
      // Check if modal is already open
      const registerModal = document.getElementById('registerModal');
      if (registerModal && registerModal.style.display === 'block') {
        console.log('Modal is already open, ignoring click');
        return;
      }
      
      // Reset any existing modal state
      if (registerModal) {
        registerModal.style.display = 'none';
        registerModal.style.visibility = 'hidden';
        registerModal.style.opacity = '0';
      }
      
      // Stop validation camera when opening register modal
      stopValidationCamera();
      
      // Small delay to ensure modal is closed before opening
      setTimeout(() => {
        registerModal.style.display = 'block';
        registerModal.style.visibility = 'visible';
        registerModal.style.opacity = '1';
        
        // Initialize register button states when modal opens
        updateRegisterButtonStates();
        updateStepperStates();
        
        // Hide the Start Camera button since camera starts automatically
        const startButton = document.getElementById('btnStartRegister');
        if (startButton) {
          startButton.style.display = 'none';
        }
      }, 50);
      
      // Debug: Check if elements exist
      setTimeout(() => {
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
        
        // Start automatic registration process
        startAutoRegisterProcess();
      }, 100);
    };

    document.getElementById('btnCloseRegister').onclick = () => {
      // Button Close tidak mengikuti sequential logic karena ada di pojok kanan atas
      
      if (registerStream) {
        registerStream.getTracks().forEach(track => track.stop());
        registerStream = null;
      }
      if (burstInterval) {
        clearInterval(burstInterval);
        burstInterval = null;
      }
      
      // Stop validation camera when closing register modal
      stopValidationCamera();
      
      // Reset modal state
      const registerVideo = document.getElementById('registerVideo');
      const registerCapturedImage = document.getElementById('registerCapturedImage');
      const cameraPlaceholder = document.getElementById('cameraPlaceholder');
      
      registerVideo.style.display = 'none';
      registerCapturedImage.style.display = 'none';
      cameraPlaceholder.style.display = 'flex';
      
      // Reset button states
      document.getElementById('btnCapturePhoto').disabled = true;
      document.getElementById('btnUpdatePhoto').disabled = true;
      document.getElementById('btnResetPhoto').disabled = true;
      document.getElementById('btnBurstCapture').disabled = true;
      
      document.getElementById('registerProgress').textContent = '';
      // Force close modal
      const registerModal = document.getElementById('registerModal');
      if (registerModal) {
        console.log('Closing modal element:', registerModal);
        registerModal.style.display = 'none';
        registerModal.style.visibility = 'hidden';
        registerModal.style.opacity = '0';
        console.log('Modal closed successfully');
      } else {
        console.error('Modal element not found!');
      }
      
      // Reset register steps when closing modal
      resetRegisterSteps();
    };

    document.getElementById('btnStartRegister').onclick = async () => {
      if (currentRegisterStep !== 1) {
        console.log('Button 1 can only be clicked when current step is 1');
        return;
      }
      
      try {
        console.log('Starting camera...');
        document.getElementById('registerProgress').textContent = 'Starting camera...';
        
        // Stop validation camera if running
        stopValidationCamera();
        
        // Stop any existing stream first
        if (registerStream) {
          registerStream.getTracks().forEach(track => track.stop());
          registerStream = null;
        }
        
        // Request camera with mobile-optimized constraints
        registerStream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 480 },
            height: { ideal: 640 },
            frameRate: { ideal: 15 }
          }
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
        
        // Burst capture button remains disabled until Update Photo success
        
        // Disable the start camera button since camera is now running
        const startButton = document.getElementById('btnStartRegister');
        startButton.disabled = true;
        startButton.style.opacity = '0.5';
        startButton.style.cursor = 'not-allowed';
        
        // Update stepper to show step 1 completed and step 2 active
        currentRegisterStep = 2; // Move to step 2 after camera starts
        updateStepperStates();
        
      } catch (e) {
        console.error('Camera error:', e);
        document.getElementById('registerProgress').textContent = 'Camera error: ' + e.message;
      }
    };

    // Burst Capture button (Step 2) - removed duplicate handler

    // Capture Photo button in modal - handled by unified function above

    // Reset Photo button in modal - handled by unified function above

    // Update Photo button in modal - handled by unified function above

    document.getElementById('btnBurstCapture').onclick = async () => {
      if (!registerStream) {
        console.error('No register stream available for burst capture');
        return;
      }
      
      const registerVideo = document.getElementById('registerVideo');
      const registerCanvas = document.getElementById('registerCanvas');
      const progress = document.getElementById('registerProgress');
      
      console.log('Starting burst capture with video stream:', registerStream);
      console.log('Video element:', registerVideo);
      console.log('Video display:', registerVideo ? registerVideo.style.display : 'not found');
      
      let capturedFrames = [];
      
      // Start burst capture immediately without countdown
      progress.textContent = 'Starting burst capture...';
      document.getElementById('btnBurstCapture').disabled = true;
      
      // Start burst capture immediately
      startBurstCapture();
      
      function startBurstCapture() {
        // Create prominent frame capture notification
        const frameNotification = document.createElement('div');
        frameNotification.id = 'frameCaptureNotification';
        frameNotification.style.cssText = `
          position: fixed;
          top: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: linear-gradient(135deg, #4CAF50, #45a049);
          color: white;
          padding: 20px 30px;
          border-radius: 15px;
          font-size: 20px;
          font-weight: bold;
          text-align: center;
          box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
          z-index: 10000;
          animation: slideDown 0.5s ease-out;
          border: 2px solid #fff;
          min-width: 300px;
        `;
        
        // Add slide animation
        const frameStyle = document.createElement('style');
        frameStyle.textContent = `
          @keyframes slideDown {
            from { transform: translateX(-50%) translateY(-100px); opacity: 0; }
            to { transform: translateX(-50%) translateY(0); opacity: 1; }
          }
          @keyframes slideUp {
            from { transform: translateX(-50%) translateY(0); opacity: 1; }
            to { transform: translateX(-50%) translateY(-100px); opacity: 0; }
          }
          @keyframes framePulse {
            0% { transform: translateX(-50%) scale(1); }
            50% { transform: translateX(-50%) scale(1.02); }
            100% { transform: translateX(-50%) scale(1); }
          }
        `;
        document.head.appendChild(frameStyle);
        
        frameNotification.innerHTML = `
          <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
            <div style="font-size: 24px;">📸</div>
            <div>
              <div style="font-size: 16px; margin-bottom: 5px;">CAPTURING FRAMES</div>
              <div id="frameCount" style="font-size: 28px; font-weight: 900;">0 / 25</div>
            </div>
            <div style="font-size: 24px;">📸</div>
          </div>
          <div style="margin-top: 10px; font-size: 14px; opacity: 0.9;">Please stay still and look at the camera</div>
        `;
        
        document.body.appendChild(frameNotification);
        
        progress.textContent = 'Capturing photos... (5 seconds)';
        
        burstInterval = setInterval(() => {
          // Create canvas with proper dimensions matching video
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = registerVideo.videoWidth;
          tempCanvas.height = registerVideo.videoHeight;
          
          const ctx = tempCanvas.getContext('2d');
          ctx.drawImage(registerVideo, 0, 0, tempCanvas.width, tempCanvas.height);
          const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
          capturedFrames.push(dataUrl);
          
          // Update frame count with animation
          const frameCount = document.getElementById('frameCount');
          if (frameCount) {
            frameCount.textContent = `${capturedFrames.length} / 25`;
            frameNotification.style.animation = 'framePulse 0.3s ease-out';
            setTimeout(() => {
              frameNotification.style.animation = '';
            }, 300);
          }
          
          progress.textContent = `Photo ${capturedFrames.length} captured...`;
        }, 200); // Take photo every 200ms
        
        setTimeout(async () => {
          clearInterval(burstInterval);
          burstInterval = null;
          
          // Remove frame notification
          const frameNotification = document.getElementById('frameCaptureNotification');
          if (frameNotification) {
            document.body.removeChild(frameNotification);
          }
          
          // Show completion notification
          const completionNotification = document.createElement('div');
          completionNotification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            padding: 20px 30px;
            border-radius: 15px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.4);
            z-index: 10000;
            animation: slideDown 0.5s ease-out;
            border: 2px solid #fff;
            min-width: 300px;
          `;
          completionNotification.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
              <div class="processing-animation">
                <div class="processing-dots">
                  <div class="dot"></div>
                  <div class="dot"></div>
                  <div class="dot"></div>
                </div>
              </div>
              <div style="color: #fff; font-size: 16px; font-weight: 500;">Processing ${capturedFrames.length} frames...</div>
            </div>
            <style>
              .processing-animation {
                display: flex;
                align-items: center;
                justify-content: center;
              }
              .processing-dots {
                display: flex;
                gap: 4px;
                align-items: center;
              }
              .dot {
                width: 8px;
                height: 8px;
                background: #fff;
                border-radius: 50%;
                animation: processingBounce 1.4s ease-in-out infinite both;
              }
              .dot:nth-child(1) { animation-delay: -0.32s; }
              .dot:nth-child(2) { animation-delay: -0.16s; }
              .dot:nth-child(3) { animation-delay: 0s; }
              @keyframes processingBounce {
                0%, 80%, 100% {
                  transform: scale(0.8);
                  opacity: 0.5;
                }
                40% {
                  transform: scale(1.2);
                  opacity: 1;
                }
              }
            </style>
          `;
          document.body.appendChild(completionNotification);
          
          progress.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px; justify-content: center;">
              <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <div>Sending photos for face encoding...</div>
            </div>
            <style>
              .loading-dots {
                display: inline-flex;
                gap: 4px;
                align-items: center;
              }
              .loading-dots span {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #4ca7e5;
                animation: loadingBounce 1.4s ease-in-out infinite both;
              }
              .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
              .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
              .loading-dots span:nth-child(3) { animation-delay: 0s; }
              @keyframes loadingBounce {
                0%, 80%, 100% {
                  transform: scale(0.8);
                  opacity: 0.5;
                }
                40% {
                  transform: scale(1.2);
                  opacity: 1;
                }
              }
            </style>
          `;
          
          // Update stepper to show burst capture completed
          updateStepperStates();
          
          // Send frames to server for encoding
          const r = await fetch('/api/register_face', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames: capturedFrames })
          });
          const j = await r.json();
          
          // Remove completion notification
          if (completionNotification) {
            document.body.removeChild(completionNotification);
          }
          
          if (j.ok) {
            // Remove the loading notification
            if (completionNotification && document.body.contains(completionNotification)) {
              document.body.removeChild(completionNotification);
            }
            
            // Show success popup
            const successNotification = document.createElement('div');
            successNotification.id = 'faceEncodingSuccess';
            successNotification.style.cssText = `
              position: fixed;
              top: 20px;
              left: 50%;
              transform: translateX(-50%);
              background: linear-gradient(135deg, #28a745, #20c997);
              color: white;
              padding: 20px 30px;
              border-radius: 12px;
              font-size: 16px;
              font-weight: 600;
              text-align: center;
              box-shadow: 0 8px 32px rgba(40, 167, 69, 0.4);
              z-index: 10000;
              animation: slideDown 0.5s ease-out;
              border: 2px solid #fff;
              min-width: 300px;
            `;
            successNotification.innerHTML = `
              <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                <div style="font-size: 24px;">✅</div>
                <div>Face encoding successful!</div>
                <div style="font-size: 24px;">✅</div>
              </div>
            `;
            document.body.appendChild(successNotification);
            
            progress.textContent = 'Face recognition registered successfully!';
            // Burst Capture remains disabled until Update Photo success
            
            // Mark all steps as completed after burst capture
            console.log('Burst capture completed, marking all steps as completed');
            updateStepperToCompleted();
            
            // Disable all buttons since process is complete
            const captureButton = document.getElementById('btnCapturePhoto');
            const updateButton = document.getElementById('btnUpdatePhoto');
            const burstButton = document.getElementById('btnBurstCapture');
            
            if (captureButton) {
              captureButton.disabled = true;
              captureButton.style.opacity = '0.5';
              captureButton.style.cursor = 'not-allowed';
            }
            
            if (updateButton) {
              updateButton.disabled = true;
              updateButton.style.opacity = '0.5';
              updateButton.style.cursor = 'not-allowed';
            }
            
            if (burstButton) {
              burstButton.disabled = true;
              burstButton.style.opacity = '0.5';
              burstButton.style.cursor = 'not-allowed';
            }
            
            console.log('All buttons disabled - process complete');
            
            // Mark face recognition as completed and update roadmap
            sessionStorage.setItem('face_registered', 'true');
            updateProgressRoadmap();
            
            // Auto remove success popup after 3 seconds
            setTimeout(() => {
              if (successNotification && document.body.contains(successNotification)) {
                successNotification.style.animation = 'slideUp 0.5s ease-out';
                setTimeout(() => {
                  if (document.body.contains(successNotification)) {
                    document.body.removeChild(successNotification);
                  }
                }, 500);
              }
            }, 3000);
            
            // Show completion message instead of auto-advancing
            progress.textContent = 'Photo updated successfully! Registration complete.';
          } else {
            // Remove the loading notification
            if (completionNotification && document.body.contains(completionNotification)) {
              document.body.removeChild(completionNotification);
            }
            
            progress.textContent = 'Error: ' + j.error;
            // Burst Capture remains disabled until Update Photo success
          }
        }, 5000);
      }
    };
  </script>
  
  <!-- Footer -->
  <footer>
    <p>© <span id="currentYearRetake"></span> FTL IT Developer. All rights lol.</p>
  </footer>
  
  <!-- Dynamic Year Script -->
  <script>
    // Set current year dynamically for retake page
    document.getElementById('currentYearRetake').textContent = new Date().getFullYear();
  </script>
</body>
</html>
"""
# Setelah Login pasti masuk sini END



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


@app.route("/")
@limiter.limit("10 per minute")  # Limit access to login page
def login():
    return render_template_string(LOGIN_HTML)


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

@app.route("/retake")
def retake():
    # Check if user is logged in
    token = session.get('gym_token')
    if not token:
        # Redirect to login page if not logged in
        return redirect(url_for('login'))
    
    # Force refresh profile data to get latest photo
    # Clear cached profile data to ensure fresh data is fetched
    session.pop('profile_data', None)
    
    return render_template_string(RETAKE_HTML)


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


@app.route("/api/retake_login", methods=["POST"])
def api_retake_login():
    data = request.get_json(force=True)
    email = data.get("email")
    password = data.get("password")
    
    if not email or not password:
        return jsonify({"ok": False, "error": "email and password required"}), 400
    
    # Get identifiers for multi-user rate limiting
    user_id = get_user_identifier(email=email)
    branch_id = get_branch_identifier()
    client_id = get_client_identifier()
    
    # Check branch capacity
    if not check_branch_capacity(branch_id):
        return jsonify({
            "ok": False, 
            "error": f"Branch is at maximum capacity ({MAX_USERS_PER_BRANCH} users). Please try again later."
        }), 429
    
    # Check if login is blocked due to brute force attempts (per user)
    if is_login_blocked(user_id):
        remaining_time = get_remaining_lockout_time(user_id)
        return jsonify({
            "ok": False, 
            "error": f"Too many failed login attempts for this user. Please try again in {remaining_time} seconds."
        }), 429

    result = gym_login_with_email(email, password)
    if not result:
        # Increment failed login attempts (per user)
        attempts = increment_login_attempts(user_id)
        remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts
        
        if remaining_attempts <= 0:
            remaining_time = get_remaining_lockout_time(user_id)
            error_msg = f"Too many failed login attempts for this user. Please try again in {remaining_time} seconds."
        else:
            error_msg = f"Login failed. {remaining_attempts} attempts remaining for this user."
        
        return jsonify({"ok": False, "error": error_msg})
    
    # Reset login attempts on successful login
    reset_login_attempts(user_id)
    
    # Add user to branch tracking
    add_user_to_branch(branch_id, user_id)
    
    token = result.get("token")
    session['gym_token'] = token
    session['user_id'] = user_id
    session['branch_id'] = branch_id
    
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


@app.route("/api/compare_with_profile", methods=["POST"])
def api_compare_with_profile():
    token = session.get('gym_token')
    if not token:
        print("DEBUG: No token found in session for compare_with_profile")
        return jsonify({"ok": False, "error": "Not logged in. Please login first."}), 401

    print(f"DEBUG: compare_with_profile called with token: {token[:20]}...")

    # Try to get profile from session first, then from API if needed
    profile = session.get('profile_data')
    if not profile:
        print("DEBUG: No profile in session, fetching from API")
        profile = gym_get_profile(token)
        if profile:
            session['profile_data'] = profile  # Cache for future use
            print("DEBUG: Profile fetched and cached in session")
        else:
            print("DEBUG: Failed to fetch profile from API")
    
    if not profile:
        return jsonify({"ok": False, "error": "Failed to fetch profile"})
    
    print(f"DEBUG: Profile data type: {type(profile)}, keys: {list(profile.keys()) if isinstance(profile, dict) else 'Not a dict'}")

    image_b64 = request.json.get("image_b64")
    bgr_live = b64_to_bgr(image_b64)
    if bgr_live is None:
        return jsonify({"ok": False, "error": "Invalid image"}), 400

    # Fetch profile photo and embed
    if not isinstance(profile, dict):
        return jsonify({"ok": False, "error": f"Profile data is not a dictionary, got: {type(profile).__name__}"})
    
    profile_url = profile.get("memberphoto")
    if not profile_url:
        print("DEBUG: No memberphoto in profile data")
        return jsonify({"ok": False, "error": "No profile photo on GymMaster"})
    
    print(f"DEBUG: Profile photo URL: {profile_url}")

    try:
        print(f"DEBUG: Attempting to download profile photo from: {profile_url}")
        r = requests.get(profile_url, timeout=15)
        r.raise_for_status()
        im = np.frombuffer(r.content, np.uint8)
        bgr_prof = cv2.imdecode(im, cv2.IMREAD_COLOR)
        
        if bgr_prof is None:
            print(f"DEBUG: Failed to decode profile photo image")
            return jsonify({"ok": False, "error": "Failed to decode profile photo"})
            
        print(f"DEBUG: Successfully downloaded and decoded profile photo, shape: {bgr_prof.shape}")
    except requests.exceptions.Timeout:
        print(f"DEBUG: Timeout downloading profile photo from: {profile_url}")
        return jsonify({"ok": False, "error": "Timeout downloading profile photo"})
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Request error downloading profile photo: {e}")
        return jsonify({"ok": False, "error": f"Network error downloading profile photo: {str(e)}"})
    except Exception as e:
        print(f"DEBUG: Unexpected error downloading profile photo: {e}")
        return jsonify({"ok": False, "error": f"Failed to download profile photo: {str(e)}"})

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

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"ok": False, "error": "No JSON data provided"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid JSON data: {str(e)}"}), 400
    
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
    try:
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
            print(f"DEBUG: Processing frame {i+1}/{len(frames)}")
            try:
                bgr = b64_to_bgr(frame_b64)
                if bgr is not None:
                    print(f"DEBUG: Frame {i+1}: Image decoded successfully, shape: {bgr.shape}")
                    emb = extract_embedding(bgr)
                    if emb is not None:
                        embeddings.append(emb)
                        print(f"DEBUG: Frame {i+1}: Face detected and embedding extracted, shape: {emb.shape}")
                    else:
                        print(f"DEBUG: Frame {i+1}: No face detected")
                else:
                    print(f"DEBUG: Frame {i+1}: Invalid image")
            except Exception as e:
                print(f"DEBUG: Error processing frame {i+1}: {e}")
                import traceback
                traceback.print_exc()

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
            
            # Convert embedding to NPY float16 format for storage efficiency
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, avg_embedding.astype(np.float16))
            embedding_npy = npy_buffer.getvalue()
            
            # Check if user already exists by email in profile data
            # We need to find the member by email from GymMaster profile
            cur.execute("SELECT id FROM member WHERE email = %s", (user_email,))
            existing = cur.fetchone()
            
            # Ensure all results are consumed to avoid "Unread result found" error
            try:
                while cur.nextset():
                    pass
            except:
                pass
            
            if existing:
                # Update existing record
                cur.execute(
                    "UPDATE member SET enc = %s WHERE id = %s",
                    (embedding_npy, existing[0])
                )
                print(f"DEBUG: Updated existing record for {user_name} (Email: {user_email}, ID: {existing[0]})")
            else:
                # Insert new record with member_id from profile
                cur.execute(
                    "INSERT INTO member (member_id, first_name, last_name, email, enc) VALUES (%s, %s, %s, %s, %s)",
                    (profile.get("id", 0), profile.get("firstname", ""), profile.get("surname", ""), user_email, embedding_npy)
                )
                print(f"DEBUG: Created new record for {user_name} (Email: {user_email})")
            
            # Ensure all results are consumed to avoid "Unread result found" error
            try:
                while cur.nextset():
                    pass
            except:
                pass
            
            conn.commit()
            cur.close()
            conn.close()
            
            # Clear cache to force reload
            invalidate_member_cache()
            reload_member_cache()
            
            return jsonify({
                "ok": True, 
                "message": f"Face recognition registered successfully for {user_email}",
                "frames_processed": len(frames),
                "faces_detected": len(embeddings)
            })
        
        except Exception as e:
            print(f"DEBUG: Database error: {e}")
            return jsonify({"ok": False, "error": f"Database error: {str(e)}"})
        
    except Exception as e:
        print(f"DEBUG: Unexpected error in register_face: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Unexpected error: {str(e)}"}), 500


# -------------------- Main --------------------
if __name__ == "__main__":
    startup_optimization()  # Now all functions are defined
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
