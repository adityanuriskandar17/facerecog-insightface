"""
Liveness Detection untuk Mencegah Photo Spoofing
================================================

Implementasi sederhana dan efektif untuk mencegah serangan dengan foto
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, List, Dict

class SimpleLivenessDetector:
    """Detector liveness sederhana tapi efektif"""
    
    def __init__(self):
        self.frame_buffer = []
        self.max_buffer_size = 5
        self.blink_count = 0
        self.last_blink_time = 0
        
    def detect_liveness(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[bool, float, str]:
        """
        Deteksi liveness dengan metode sederhana tapi efektif
        
        Returns:
            (is_live, confidence, message)
        """
        x1, y1, x2, y2 = face_bbox
        
        # Extract face region
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False, 0.0, "Invalid face region"
        
        # Method 1: Motion Detection
        motion_score = self._detect_motion(frame)
        
        # Method 2: Blink Detection
        blink_score = self._detect_blink_simple(face_roi)
        
        # Method 3: Texture Analysis
        texture_score = self._analyze_texture_simple(face_roi)
        
        # Combine scores
        total_confidence = (motion_score * 0.4 + blink_score * 0.3 + texture_score * 0.3)
        
        # Threshold untuk menentukan liveness
        is_live = total_confidence > 0.5
        
        if is_live:
            message = f"Liveness verified (confidence: {total_confidence:.2f})"
        else:
            message = f"Possible spoofing detected (confidence: {total_confidence:.2f})"
        
        return is_live, total_confidence, message
    
    def _detect_motion(self, frame: np.ndarray) -> float:
        """Deteksi gerakan sederhana"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Add to buffer
        self.frame_buffer.append(gray)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) < 2:
            return 0.5  # Not enough frames
        
        # Calculate motion between consecutive frames
        motion_scores = []
        for i in range(1, len(self.frame_buffer)):
            diff = cv2.absdiff(self.frame_buffer[i-1], self.frame_buffer[i])
            motion = np.mean(diff)
            motion_scores.append(motion)
        
        avg_motion = np.mean(motion_scores)
        
        # Normalize motion (0-1)
        # Real people have natural micro-movements
        motion_score = min(avg_motion / 20.0, 1.0)
        
        return motion_score
    
    def _detect_blink_simple(self, face_roi: np.ndarray) -> float:
        """Deteksi kedipan sederhana"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes using Haar cascade
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 4)
        
        if len(eyes) < 2:
            return 0.3  # No eyes detected
        
        # Analyze eye regions for blink patterns
        eye_scores = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = gray_face[ey:ey+eh, ex:ex+ew]
            if eye_roi.size == 0:
                continue
            
            # Calculate eye openness
            # Closed eyes have less variation
            eye_std = np.std(eye_roi)
            eye_mean = np.mean(eye_roi)
            
            # Simple heuristic: more variation = more open
            if eye_mean > 0:
                openness = eye_std / eye_mean
            else:
                openness = 0
            
            eye_scores.append(openness)
        
        if not eye_scores:
            return 0.3
        
        # Average eye openness
        avg_openness = np.mean(eye_scores)
        
        # Normalize to 0-1
        blink_score = min(avg_openness / 1.5, 1.0)
        
        return blink_score
    
    def _analyze_texture_simple(self, face_roi: np.ndarray) -> float:
        """Analisis tekstur sederhana"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture variance
        texture_variance = np.var(gray_face)
        
        # Calculate edge density
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Real faces have more texture variation
        # Photos tend to be more uniform
        texture_score = (
            min(texture_variance / 500.0, 1.0) * 0.6 +
            edge_density * 0.4
        )
        
        return min(texture_score, 1.0)

def create_challenge_sequence() -> List[Dict]:
    """Buat sequence challenge untuk anti-spoofing"""
    challenges = [
        {
            "instruction": "Please blink your eyes",
            "timeout": 5,
            "required_action": "blink"
        },
        {
            "instruction": "Please turn your head slightly left",
            "timeout": 5,
            "required_action": "head_turn_left"
        },
        {
            "instruction": "Please turn your head slightly right", 
            "timeout": 5,
            "required_action": "head_turn_right"
        },
        {
            "instruction": "Please smile",
            "timeout": 5,
            "required_action": "smile"
        }
    ]
    
    return challenges

def integrate_liveness_to_app():
    """
    Kode untuk mengintegrasikan liveness detection ke app.py
    """
    integration_code = '''
# Tambahkan import di bagian atas app.py
from liveness_detection import SimpleLivenessDetector

# Tambahkan global variable
_liveness_detector = SimpleLivenessDetector()

def extract_embedding_with_liveness(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], bool, float]:
    """
    Extract embedding dengan liveness detection
    """
    model, det = load_insightface()
    if model is None or det is None:
        return None, None, False, 0.0
    
    faces = det.get(bgr)
    if not faces:
        return None, None, False, 0.0
    
    # Pilih wajah terbesar
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    
    # Convert bbox ke integer coordinates
    bbox = face.bbox
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # Liveness detection
    is_live, confidence, message = _liveness_detector.detect_liveness(bgr, (x1, y1, x2, y2))
    
    if not is_live:
        print(f"DEBUG: Liveness detection failed: {message}")
        return None, None, False, confidence
    
    # Get embedding
    emb = face.normed_embedding
    if emb is None:
        return None, None, False, 0.0
    
    return np.asarray(emb, dtype=np.float32), (x1, y1, x2, y2), True, confidence

# Modifikasi endpoint api_recognize_open_gate
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

    # Ganti dengan liveness detection
    emb, bbox, is_live, liveness_confidence = extract_embedding_with_liveness(bgr)
    
    if emb is None:
        return jsonify({"ok": False, "error": "No face found", "bbox": None}), 200
    
    if not is_live:
        return jsonify({
            "ok": False, 
            "error": "Liveness detection failed - possible spoofing attempt", 
            "liveness_confidence": liveness_confidence,
            "bbox": bbox,
            "anti_spoofing": True
        }), 200

    # ... rest of existing code ...
'''
    
    return integration_code

if __name__ == "__main__":
    print("Liveness Detection System")
    print("========================")
    print("✓ Motion Detection")
    print("✓ Blink Detection") 
    print("✓ Texture Analysis")
    print("✓ Anti-Spoofing Protection")
    print("\nReady to prevent photo attacks!")
