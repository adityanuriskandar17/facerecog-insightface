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
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import cv2
import mysql.connector
import numpy as np
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

GYM_API_KEY = os.getenv("GYM_API_KEY", "")
GYM_BASE_URL = os.getenv("GYM_BASE_URL", "")
GYM_LOGIN_URL = os.getenv("GYM_LOGIN_URL", "")
GYM_PROFILE_URL = os.getenv("GYM_PROFILE_URL", "")
GYM_GATE_URL = os.getenv("GYM_GATE_URL", "")
CHECKIN_ENABLED = os.getenv("CHECKIN_ENABLED", "True").lower() == "true"

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


def fetch_member_encodings() -> List[MemberEnc]:
    """
    Expected table structure (example):
      member (id BIGINT PK, gym_member_id BIGINT, email VARCHAR, enc LONGTEXT)
    where `enc` stores a JSON array of floats (length 512) or a base64 npy blob.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    try:
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
    finally:
        cur.close()
        conn.close()


# Cache encodings in memory for faster search
_MEMBER_CACHE: List[MemberEnc] = []


def ensure_cache_loaded():
    global _MEMBER_CACHE
    if not _MEMBER_CACHE:
        print("DEBUG: Loading member encodings from database...")
        _MEMBER_CACHE = fetch_member_encodings()
        print(f"DEBUG: Loaded {len(_MEMBER_CACHE)} member encodings")
        for member in _MEMBER_CACHE:
            print(f"DEBUG: Member {member.member_pk}: {member.email} (gym_id: {member.gym_member_id})")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_best_match(query_vec: np.ndarray) -> Tuple[Optional[MemberEnc], float, float]:
    """Return (best_member, best_score, second_best_score)."""
    ensure_cache_loaded()
    if not _MEMBER_CACHE:
        return None, 0.0, 0.0
    # Query already normalized
    sims = [(m, float(np.dot(query_vec, m.enc))) for m in _MEMBER_CACHE]
    sims.sort(key=lambda x: x[1], reverse=True)
    best = sims[0]
    second = sims[1] if len(sims) > 1 else (None, 0.0)
    return best[0], best[1], second[1]


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
    try:
        r = requests.post(GYM_GATE_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("error") is None:
            return data["result"]["response"]
    except Exception:
        return None
    return None


def gym_get_profile(token: str) -> Optional[Dict]:
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
                return result
            elif isinstance(result, dict) and len(result) > 1:  # Has multiple fields, likely profile data
                print(f"DEBUG: Found profile data with keys: {list(result.keys())}")  # Debug logging
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
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; margin: 24px; }
    .row { display: flex; gap: 24px; align-items: flex-start; }
    video, canvas, img { width: 420px; height: 315px; background: #111; border-radius: 12px; object-fit: cover; }
    button { padding: 10px 16px; border-radius: 10px; border: 1px solid #ddd; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .ok { color: #0a7; }
    .warn { color: #b70; }
    .err { color: #c31; }n
    .pill { padding: 2px 8px; border: 1px solid #999; border-radius: 999px; font-size: 12px; }
    .card { border: 1px solid #eee; padding: 16px; border-radius: 12px; }
    .muted { color: #777; }
    .field { display: inline-flex; gap: 8px; align-items: center; }
    .stack { display: grid; gap: 12px; }
  </style>
</head>
<body>
    <div style="display: flex; justify-content: center; align-items: center; min-height: 100vh; flex-direction: column;">
      <h1 style="margin-bottom: 20px;">FTL Face Gate</h1>
      <div style="margin-bottom: 20px;">
        Door ID: <span id="doorid" class="pill">19456</span>
          </div>
      
      <!-- Single Camera in Center -->
      <div style="position: relative; display: inline-block; margin-bottom: 20px;">
        <video id="video" autoplay playsinline muted style="width: 640px; height: 480px; background: #f0f0f0; border-radius: 12px; object-fit: cover; border: 2px solid #ddd;"></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0; pointer-events: none; border-radius: 12px; width: 640px; height: 480px; z-index: 10; background: transparent;"></canvas>
        <div id="cameraStatus" style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px;">Camera inactive</div>
        </div>
      
      <!-- Control Buttons -->
      <div style="margin-bottom: 20px;">
        <button id="btnStart" disabled style="margin: 5px; padding: 10px 20px; border: none; border-radius: 5px; background: #007bff; color: white; cursor: pointer;">Start Camera</button>
        <button id="btnStop" disabled style="margin: 5px; padding: 10px 20px; border: none; border-radius: 5px; background: #dc3545; color: white; cursor: pointer;">Stop Camera</button>
        <button id="btnRetake" onclick="location.href='/login'" style="margin: 5px; padding: 10px 20px; border: none; border-radius: 5px; background: #28a745; color: white; cursor: pointer;">Register Face</button>
        <button id="btnDebug" onclick="debugCamera()" style="margin: 5px; padding: 10px 20px; border: none; border-radius: 5px; background: #ffc107; color: black; cursor: pointer;">Debug Camera</button>
        <button id="btnToggleOverlay" onclick="toggleOverlay()" style="margin: 5px; padding: 10px 20px; border: none; border-radius: 5px; background: #6f42c1; color: white; cursor: pointer;">Hide Overlay</button>
      </div>
      
      <!-- Detection Result - Just Name -->
      <div id="detectionResult" style="padding: 20px; margin: 20px 0; border-radius: 10px; background: #f8f9fa; border: 1px solid #dee2e6; min-height: 80px; text-align: center;">
        <div id="detectedName" style="font-size: 32px; font-weight: bold; color: #333; margin-bottom: 10px;">No face detected</div>
        <div id="detectionStatus" style="font-size: 18px; color: #666;">Camera inactive</div>
      </div>
      
      <div style="color: #666; font-size: 14px;">
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

    function setLog(obj) {
      console.log(typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2));
    }
    
    function drawBoundingBox(x, y, width, height, color = '#00FF00', label = '') {
      if (!overlay) return;
      
      const ctx = overlay.getContext('2d');
      const videoWidth = video.videoWidth || 640;
      const videoHeight = video.videoHeight || 480;
      
      // Set canvas size to match video
      overlay.width = videoWidth;
      overlay.height = videoHeight;
      overlay.style.width = '640px';
      overlay.style.height = '480px';
      
      // Clear canvas with transparent background
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      
      if (x && y && width && height) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        
        if (label) {
          ctx.fillStyle = color;
          ctx.font = 'bold 16px Arial';
          ctx.fillText(label, x, Math.max(y - 10, 20));
        }
      }
    }

    function clearBoundingBox() {
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function setButtons(running) {
      btnStart.disabled = running || !doorid;
      btnStop.disabled = !running;
    }

    function updateDetectionDisplay(name, status, confidence = null) {
      detectedName.textContent = name || 'No face detected';
      detectionStatus.textContent = status;
      
      if (name && confidence) {
        detectedName.style.color = confidence > 0.6 ? '#4CAF50' : '#FF9800';
        detectionStatus.textContent = `${status} (${Math.round(confidence * 100)}% confidence)`;
      } else {
        detectedName.style.color = '#333';
      }
    }

    async function startCam() {
      try {
        console.log('Requesting camera access...');
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
        console.log('Camera stream obtained:', stream);
        
        video.srcObject = stream;
        console.log('Video srcObject set');
        
        // Wait for video to load
        video.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
          cameraStatus.textContent = 'Camera active';
          cameraStatus.style.background = 'rgba(0,150,0,0.8)';
        };
        
        video.oncanplay = () => {
          console.log('Video can play');
          cameraStatus.textContent = 'Camera streaming';
          cameraStatus.style.background = 'rgba(0,150,0,0.8)';
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
        canvas.width = 640;
        canvas.height = 480;
        console.log('Canvas created');
        
        setButtons(true);
        setLog('Camera started.');
        updateDetectionDisplay(null, 'Camera active - detecting faces...');
        
        // Start automatic face recognition every 2 seconds
        recognitionInterval = setInterval(performRecognition, 2000);
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
      setButtons(false);
      setLog('Camera stopped.');
      updateDetectionDisplay(null, 'Camera inactive');
      cameraStatus.textContent = 'Camera inactive';
      cameraStatus.style.background = 'rgba(0,0,0,0.7)';
    }

    async function performRecognition() {
      if (!stream || !doorid || !canvas) return;
      
      const now = Date.now();
      if (now - lastRecognitionTime < 1500) return; // Throttle to prevent too frequent calls
      lastRecognitionTime = now;
      
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
          updateDetectionDisplay(name, 'Face recognized', confidence);
          
          // Draw bounding box for recognized face
          if (j.bbox && j.bbox.length === 4) {
            const [x1, y1, x2, y2] = j.bbox;
            drawBoundingBox(x1, y1, x2-x1, y2-y1, '#4CAF50', name);
          } else {
            // Draw a default box if no bbox
            drawBoundingBox(100, 100, 200, 200, '#4CAF50', name);
          }
          
          // Auto-open gate if matched
          if (j.gate && !j.gate.error) {
            updateDetectionDisplay(name, 'Gate opened successfully!', confidence);
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
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Arial; margin: 24px; }
    .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
    video, img, canvas { width: 360px; height: 360px; object-fit: cover; border-radius: 12px; background: #111; }
    .card { border: 1px solid #eee; padding: 16px; border-radius: 12px; }
    button { padding: 10px 16px; border-radius: 10px; border: 1px solid #ddd; cursor: pointer; }
    input { padding: 8px 12px; border: 1px solid #ccc; border-radius: 8px; width: 280px; }
    .stack { display: grid; gap: 12px; }
  </style>
</head>
<body>
  <h1>Profile & Face Registration</h1>
  <div class="row">
    <div class="card stack">
      <h3>User Profile</h3>
      <div id="profileInfo" style="text-align: center; padding: 20px;">
        <div id="loadingProfile">Loading profile...</div>
      </div>
      <div style="text-align: center; margin-top: 15px;">
        <button id="btnLogout" onclick="logout()" style="padding: 8px 16px; background: #dc3545; color: white; border: none; border-radius: 5px; cursor: pointer;">Logout</button>
      </div>
      <pre id="log"></pre>
    </div>
    <div class="card stack">
      <h3>Profile Photo</h3>
      <img id="profile" alt="profile" />
    </div>
    <div class="card stack">
      <h3>Capture & Compare</h3>
      <video id="video" autoplay playsinline muted></video>
      <button id="btnStart">Start</button>
      <button id="btnSnap" disabled>Capture & Compare</button>
      <button id="btnRegister" style="background: #4CAF50; color: white;">Daftarkan Face Recognition</button>
      <canvas id="canvas" width="640" height="480" style="display:none"></canvas>
      <pre id="out"></pre>
    </div>
  </div>
  
  <!-- Modal for Face Registration -->
  <div id="registerModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; padding: 24px; border-radius: 12px; max-width: 600px; width: 90%;">
      <h2>Daftarkan Face Recognition</h2>
      <div style="text-align: center; margin: 20px 0;">
        <video id="registerVideo" autoplay playsinline muted style="width: 400px; height: 300px; background: #111; border-radius: 8px;"></video>
        <div style="margin: 16px 0;">
          <button id="btnStartRegister" style="padding: 12px 24px; margin: 8px; background: #2196F3; color: white; border: none; border-radius: 8px; cursor: pointer;">Mulai Kamera</button>
          <button id="btnBurstCapture" disabled style="padding: 12px 24px; margin: 8px; background: #FF9800; color: white; border: none; border-radius: 8px; cursor: pointer;">Burst Foto (5 detik)</button>
          <button id="btnCloseRegister" style="padding: 12px 24px; margin: 8px; background: #f44336; color: white; border: none; border-radius: 8px; cursor: pointer;">Tutup</button>
        </div>
        <div id="registerProgress" style="margin: 16px 0; font-size: 14px; color: #666;"></div>
        <canvas id="registerCanvas" width="640" height="480" style="display: none;"></canvas>
      </div>
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
        loadingProfile.textContent = 'Loading profile...';
        
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
          
          // Display profile info
          profileInfo.innerHTML = `
            <div style="margin-bottom: 15px;">
              <strong>Name:</strong> ${profile.fullname || 'N/A'}<br>
              <strong>Email:</strong> ${profile.email || 'N/A'}<br>
              <strong>Phone:</strong> ${profile.phonecell || 'N/A'}<br>
              <strong>Member ID:</strong> ${profile.id || 'N/A'}
            </div>
          `;
          
          // Load profile photo
          if (profile.memberphoto) {
            setLog('Loading profile photo: ' + profile.memberphoto);
            img.src = profile.memberphoto;
            img.onerror = () => {
              setLog('Error loading profile photo: ' + profile.memberphoto);
              img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzYwIiBoZWlnaHQ9IjM2MCIgdmlld0JveD0iMCAwIDM2MCAzNjAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIzNjAiIGhlaWdodD0iMzYwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xODAgMTIwQzE0OC41IDEyMCAxMjAgMTQ4LjUgMTIwIDE4MEMxMjAgMjExLjUgMTQ4LjUgMjQwIDE4MCAyNDBDMjExLjUgMjQwIDI0MCAyMTEuNSAyNDAgMTgwQzI0MCAxNDguNSAyMTEuNSAxMjAgMTgwIDEyMFoiIGZpbGw9IiM5Q0EzQUYiLz4KPHBhdGggZD0iTTE4MCAyMDBDMTY4IDIwMCAxNTggMTkwIDE1OCAxODBDMTU4IDE2NiAxNjggMTU2IDE4MCAxNTZDMjA0IDE1NiAyMTQgMTY2IDIxNCAxODBDMjE0IDE5MCAyMDQgMjAwIDE4MCAyMDBaIiBmaWxsPSIjRkZGRkZGIi8+Cjx0ZXh0IHg9IjE4MCIgeT0iMzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNjY2NjY2IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPk5vIFBob3RvPC90ZXh0Pgo8L3N2Zz4K';
            };
            img.onload = () => {
              setLog('Profile photo loaded successfully');
            };
          } else {
            setLog('No profile photo found');
            img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzYwIiBoZWlnaHQ9IjM2MCIgdmlld0JveD0iMCAwIDM2MCAzNjAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIzNjAiIGhlaWdodD0iMzYwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xODAgMTIwQzE0OC41IDEyMCAxMjAgMTQ4LjUgMTIwIDE4MEMxMjAgMjExLjUgMTQ4LjUgMjQwIDE4MCAyNDBDMjExLjUgMjQwIDI0MCAyMTEuNSAyNDAgMTgwQzI0MCAxNDguNSAyMTEuNSAxMjAgMTgwIDEyMFoiIGZpbGw9IiM5Q0EzQUYiLz4KPHBhdGggZD0iTTE4MCAyMDBDMTY4IDIwMCAxNTggMTkwIDE1OCAxODBDMTU4IDE2NiAxNjggMTU2IDE4MCAxNTZDMjA0IDE1NiAyMTQgMTY2IDIxNCAxODBDMjE0IDE5MCAyMDQgMjAwIDE4MCAyMDBaIiBmaWxsPSIjRkZGRkZGIi8+Cjx0ZXh0IHg9IjE4MCIgeT0iMzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNjY2NjY2IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPk5vIFBob3RvPC90ZXh0Pgo8L3N2Zz4K';
          }
        } else {
          profileInfo.innerHTML = '<div style="color: red;">Not logged in. Please go back to login page.</div>';
        }
      } catch (e) {
        profileInfo.innerHTML = '<div style="color: red;">Error loading profile: ' + e.message + '</div>';
        setLog('Error: ' + e.message);
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
        document.getElementById('video').srcObject = stream;
        document.getElementById('btnStart').disabled = true;
        document.getElementById('btnSnap').disabled = false;
        setOut('Camera started. Click "Capture & Compare" to take a photo.');
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

    // Face Registration Modal Controls
    document.getElementById('btnRegister').onclick = () => {
      document.getElementById('registerModal').style.display = 'block';
    };

    document.getElementById('btnCloseRegister').onclick = () => {
      document.getElementById('registerModal').style.display = 'none';
      if (registerStream) {
        registerStream.getTracks().forEach(t => t.stop());
        registerStream = null;
      }
      if (burstInterval) {
        clearInterval(burstInterval);
        burstInterval = null;
      }
    };

    document.getElementById('btnStartRegister').onclick = async () => {
      try {
        registerStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        document.getElementById('registerVideo').srcObject = registerStream;
        document.getElementById('btnBurstCapture').disabled = false;
        document.getElementById('registerProgress').textContent = 'Kamera siap. Klik "Burst Foto" untuk mulai.';
      } catch (e) {
        document.getElementById('registerProgress').textContent = 'Error: ' + e.message;
      }
    };

    document.getElementById('btnBurstCapture').onclick = async () => {
      if (!registerStream) return;
      
      const registerVideo = document.getElementById('registerVideo');
      const registerCanvas = document.getElementById('registerCanvas');
      const progress = document.getElementById('registerProgress');
      
      let capturedFrames = [];
      let countdown = 5;
      
      progress.textContent = `Burst foto dimulai dalam ${countdown} detik...`;
      document.getElementById('btnBurstCapture').disabled = true;
      
      // Countdown
      const countdownInterval = setInterval(() => {
        countdown--;
        progress.textContent = `Burst foto dimulai dalam ${countdown} detik...`;
        if (countdown <= 0) {
          clearInterval(countdownInterval);
          startBurstCapture();
        }
      }, 1000);
      
      function startBurstCapture() {
        progress.textContent = 'Mengambil foto... (5 detik)';
        
        burstInterval = setInterval(() => {
          const ctx = registerCanvas.getContext('2d');
          ctx.drawImage(registerVideo, 0, 0, registerCanvas.width, registerCanvas.height);
          const dataUrl = registerCanvas.toDataURL('image/jpeg', 0.9);
          capturedFrames.push(dataUrl);
          progress.textContent = `Foto ${capturedFrames.length} diambil...`;
        }, 200); // Take photo every 200ms
        
        setTimeout(async () => {
          clearInterval(burstInterval);
          burstInterval = null;
          progress.textContent = 'Mengirim foto untuk encoding...';
          
          // Send frames to server for encoding
          const r = await fetch('/api/register_face', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames: capturedFrames })
      });
      const j = await r.json();
          
          if (j.ok) {
            progress.textContent = 'Face recognition berhasil didaftarkan!';
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
        token = gym_login_with_memberid(best.gym_member_id)
        if token:
            session['gym_token'] = token
            gate = gym_open_gate(token, int(doorid))
            resp["gate"] = gate if gate else {"error": "Gate API failed"}
        else:
            resp["gate"] = {"error": "Login API failed"}
    else:
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
        global _MEMBER_CACHE
        _MEMBER_CACHE = []
        
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
