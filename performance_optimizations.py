u"""
Performance Optimizations for Face Recognition
==============================================

File ini berisi optimasi untuk mempercepat face recognition:
1. Preload model saat startup
2. Optimasi Redis cache
3. Database query optimization
4. Face detection optimization
"""

import os
import time
import threading
import numpy as np
import cv2
from typing import Optional, Tuple, List
import redis
import json
import pickle

# Global variables untuk optimasi
_face_rec_model = None
_face_det = None
_insightface_lock = threading.Lock()
_model_loaded = False

# Optimized Redis settings
REDIS_CACHE_TTL = 7200  # 2 hours instead of 1 hour
REDIS_PRELOAD_ENABLED = True

def preload_insightface_model():
    """
    Preload InsightFace model saat startup untuk menghindari lazy loading
    yang menyebabkan delay saat face recognition pertama kali
    """
    global _face_rec_model, _face_det, _model_loaded
    
    if _model_loaded:
        return _face_rec_model, _face_det
    
    try:
        with _insightface_lock:
            if _face_rec_model is None:
                print("DEBUG: Preloading InsightFace model at startup...")
                from insightface.app import FaceAnalysis
                
                # Gunakan model yang lebih kecil dan cepat
                _face_rec_model = FaceAnalysis(
                    name="buffalo_s",  # Model lebih kecil dari buffalo_l
                    providers=["CPUExecutionProvider"]
                )
                
                # Gunakan resolusi deteksi yang lebih kecil untuk kecepatan
                _face_rec_model.prepare(ctx_id=0, det_size=(320, 320))
                _face_det = _face_rec_model
                _model_loaded = True
                print("DEBUG: InsightFace model preloaded successfully")
                
    except Exception as e:
        print(f"DEBUG: Error preloading InsightFace model: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    return _face_rec_model, _face_det

def extract_embedding_optimized(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Optimized face embedding extraction dengan:
    1. Image resizing untuk deteksi yang lebih cepat
    2. Model preloading
    3. Optimized face selection
    """
    model, det = preload_insightface_model()
    if model is None or det is None:
        return None, None
    
    # Resize image jika terlalu besar untuk deteksi yang lebih cepat
    height, width = bgr.shape[:2]
    if width > 640:
        scale = 640 / width
        new_width = 640
        new_height = int(height * scale)
        bgr_resized = cv2.resize(bgr, (new_width, new_height))
    else:
        bgr_resized = bgr
    
    try:
        faces = det.get(bgr_resized)
        if not faces:
            return None, None
        
        # Pilih wajah terbesar dengan area terbesar
        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        
        # Dapatkan embedding yang sudah dinormalisasi
        emb = best_face.normed_embedding
        if emb is None:
            return None, None
        
        # Convert bbox ke integer coordinates
        bbox = best_face.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Scale bbox coordinates back jika image di-resize
        if width > 640:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
        
        return np.asarray(emb, dtype=np.float32), (x1, y1, x2, y2)
        
    except Exception as e:
        print(f"DEBUG: Error in extract_embedding_optimized: {e}")
        return None, None

def optimize_redis_cache():
    """
    Optimasi Redis cache dengan:
    1. TTL yang lebih panjang
    2. Compression untuk data besar
    3. Connection pooling
    """
    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=False,
            socket_connect_timeout=1,
            socket_timeout=1,
            max_connections=10  # Connection pooling
        )
        
        # Test connection
        r.ping()
        return r
    except Exception as e:
        print(f"DEBUG: Redis connection failed: {e}")
        return None

def preload_member_cache():
    """
    Preload member encodings ke cache saat startup
    untuk menghindari delay saat face recognition pertama kali
    """
    try:
        # Import dari app.py
        from app import ensure_cache_loaded, _MEMBER_CACHE
        
        print("DEBUG: Preloading member cache...")
        ensure_cache_loaded(force_refresh=True)
        
        if _MEMBER_CACHE:
            print(f"DEBUG: Preloaded {len(_MEMBER_CACHE)} members to cache")
            return True
        else:
            print("DEBUG: No members found to preload")
            return False
            
    except Exception as e:
        print(f"DEBUG: Error preloading member cache: {e}")
        return False

def optimize_database_query():
    """
    Optimasi database query dengan:
    1. Index pada kolom enc
    2. Query yang lebih efisien
    3. Connection pooling
    """
    queries = [
        # Tambahkan index pada kolom enc untuk query yang lebih cepat
        "ALTER TABLE member ADD INDEX idx_enc (enc(100))",
        
        # Optimasi query dengan WHERE clause yang lebih spesifik
        """
        SELECT id AS member_pk, member_id AS gym_member_id, 
               CONCAT(first_name, ' ', last_name) AS email, enc
        FROM member
        WHERE enc IS NOT NULL 
          AND enc != '' 
          AND LENGTH(enc) > 100
          AND member_id IS NOT NULL
        ORDER BY id
        """
    ]
    
    return queries

def benchmark_face_recognition():
    """
    Benchmark untuk mengukur performa face recognition
    """
    import time
    
    # Test image (dummy)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Benchmark model loading
    start_time = time.time()
    model, det = preload_insightface_model()
    model_load_time = time.time() - start_time
    
    # Benchmark face detection
    start_time = time.time()
    emb, bbox = extract_embedding_optimized(test_image)
    detection_time = time.time() - start_time
    
    print(f"DEBUG: Model load time: {model_load_time:.3f}s")
    print(f"DEBUG: Face detection time: {detection_time:.3f}s")
    
    return {
        "model_load_time": model_load_time,
        "detection_time": detection_time,
        "total_time": model_load_time + detection_time
    }

if __name__ == "__main__":
    print("Face Recognition Performance Optimizations")
    print("===========================================")
    
    # Preload model
    print("1. Preloading InsightFace model...")
    preload_insightface_model()
    
    # Preload cache
    print("2. Preloading member cache...")
    preload_member_cache()
    
    # Test Redis connection
    print("3. Testing Redis connection...")
    redis_conn = optimize_redis_cache()
    if redis_conn:
        print("   ✓ Redis connected successfully")
    else:
        print("   ✗ Redis connection failed")
    
    # Benchmark
    print("4. Running performance benchmark...")
    benchmark_results = benchmark_face_recognition()
    
    print("\nOptimization Summary:")
    print(f"- Model load time: {benchmark_results['model_load_time']:.3f}s")
    print(f"- Face detection time: {benchmark_results['detection_time']:.3f}s")
    print(f"- Total time: {benchmark_results['total_time']:.3f}s")
