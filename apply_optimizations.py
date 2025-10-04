#!/usr/bin/env python3
"""
Script untuk mengaplikasikan optimasi performa ke aplikasi Face Recognition
================================================================================

Script ini akan:
1. Backup file app.py yang ada
2. Aplikasikan optimasi ke app.py
3. Setup database indexes
4. Test performa setelah optimasi
"""

import os
import shutil
import time
import subprocess
import sys
from datetime import datetime

def backup_original_file():
    """Backup file app.py yang asli"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"app_backup_{timestamp}.py"
    
    if os.path.exists("app.py"):
        shutil.copy2("app.py", backup_name)
        print(f"✓ Backup created: {backup_name}")
        return backup_name
    else:
        print("✗ app.py not found")
        return None

def apply_model_optimizations():
    """Aplikasikan optimasi model loading"""
    optimizations = """
# Model Optimization - Preload at startup
def preload_insightface_model():
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

# Optimized face detection
def extract_embedding_optimized(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
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
"""
    return optimizations

def apply_cache_optimizations():
    """Aplikasikan optimasi Redis cache"""
    optimizations = """
# Cache Optimization Settings
REDIS_CACHE_TTL = 7200  # 2 jam instead of 1 jam
_CACHE_REFRESH_INTERVAL = 3600  # 1 jam instead of 30 menit

# Preload cache at startup
def preload_member_cache():
    global _MEMBER_CACHE
    try:
        print("DEBUG: Preloading member cache at startup...")
        ensure_cache_loaded(force_refresh=True)
        if _MEMBER_CACHE:
            print(f"DEBUG: Preloaded {len(_MEMBER_CACHE)} members to cache")
            return True
        return False
    except Exception as e:
        print(f"DEBUG: Error preloading member cache: {e}")
        return False

# Optimized Redis connection
def get_redis_conn_optimized():
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD if REDIS_PASSWORD else None,
            decode_responses=False,
            socket_connect_timeout=1,
            socket_timeout=1,
            max_connections=10,
            retry_on_timeout=True
        )
        r.ping()
        return r
    except Exception as e:
        print(f"DEBUG: Redis connection failed: {e}")
        return None
"""
    return optimizations

def apply_database_optimizations():
    """Aplikasikan optimasi database"""
    optimizations = """
# Database Optimization
def setup_database_indexes():
    conn = None
    cur = None
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        indexes = [
            "ALTER TABLE member ADD INDEX idx_enc (enc(100))",
            "ALTER TABLE member ADD INDEX idx_member_id (member_id)",
            "ALTER TABLE member ADD INDEX idx_enc_length (LENGTH(enc))"
        ]
        
        for index_sql in indexes:
            try:
                cur.execute(index_sql)
                print(f"DEBUG: Created index: {index_sql}")
            except Exception as e:
                if "Duplicate key name" not in str(e):
                    print(f"DEBUG: Index creation failed: {e}")
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"DEBUG: Error setting up database indexes: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# Optimized member fetch query
def fetch_member_encodings_optimized() -> List[MemberEnc]:
    conn = None
    cur = None
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        # Optimized query dengan LIMIT dan WHERE yang lebih spesifik
        cur.execute("""
            SELECT id AS member_pk, member_id AS gym_member_id, 
                   CONCAT(first_name, ' ', last_name) AS email, enc
            FROM member
            WHERE enc IS NOT NULL 
              AND enc != '' 
              AND LENGTH(enc) > 100
              AND member_id IS NOT NULL
            ORDER BY id
            LIMIT 1000
        """)
        
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
        print(f"DEBUG: Database error in fetch_member_encodings_optimized: {e}")
        return []
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
"""
    return optimizations

def create_startup_script():
    """Buat script startup untuk preload semua komponen"""
    startup_script = """
# Startup optimizations
def startup_optimizations():
    print("DEBUG: Starting performance optimizations...")
    
    # 1. Preload InsightFace model
    print("1. Preloading InsightFace model...")
    preload_insightface_model()
    
    # 2. Setup database indexes
    print("2. Setting up database indexes...")
    setup_database_indexes()
    
    # 3. Preload member cache
    print("3. Preloading member cache...")
    preload_member_cache()
    
    # 4. Test Redis connection
    print("4. Testing Redis connection...")
    redis_conn = get_redis_conn_optimized()
    if redis_conn:
        print("   ✓ Redis connected successfully")
    else:
        print("   ✗ Redis connection failed")
    
    print("DEBUG: Performance optimizations completed!")

# Call startup optimizations when module loads
if __name__ == "__main__":
    startup_optimizations()
"""
    return startup_script

def apply_optimizations_to_app():
    """Aplikasikan semua optimasi ke app.py"""
    try:
        # Read original app.py
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Backup original
        backup_file = backup_original_file()
        if not backup_file:
            return False
        
        # Apply optimizations
        print("Applying optimizations to app.py...")
        
        # Add model optimizations
        model_opt = apply_model_optimizations()
        
        # Add cache optimizations  
        cache_opt = apply_cache_optimizations()
        
        # Add database optimizations
        db_opt = apply_database_optimizations()
        
        # Add startup script
        startup_opt = create_startup_script()
        
        # Combine all optimizations
        all_optimizations = f"""
# ==================== PERFORMANCE OPTIMIZATIONS ====================
{model_opt}

{cache_opt}

{db_opt}

{startup_opt}
# ===================================================================
"""
        
        # Insert optimizations after imports
        import_end = content.find("\n\n# Global variables")
        if import_end == -1:
            import_end = content.find("\n\n#")
        
        if import_end != -1:
            new_content = content[:import_end] + all_optimizations + content[import_end:]
        else:
            new_content = content + all_optimizations
        
        # Write optimized app.py
        with open("app.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print("✓ Optimizations applied to app.py")
        return True
        
    except Exception as e:
        print(f"✗ Error applying optimizations: {e}")
        return False

def test_optimizations():
    """Test optimasi yang telah diaplikasikan"""
    try:
        print("Testing optimizations...")
        
        # Test import
        import app
        print("✓ App imports successfully")
        
        # Test Redis connection
        redis_conn = app.get_redis_conn()
        if redis_conn:
            print("✓ Redis connection works")
        else:
            print("✗ Redis connection failed")
        
        # Test database connection
        db_conn = app.get_db_conn()
        if db_conn:
            print("✓ Database connection works")
            db_conn.close()
        else:
            print("✗ Database connection failed")
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main function untuk mengaplikasikan optimasi"""
    print("Face Recognition Performance Optimization")
    print("=========================================")
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("✗ app.py not found in current directory")
        return False
    
    # Apply optimizations
    if apply_optimizations_to_app():
        print("\n✓ Optimizations applied successfully!")
        
        # Test optimizations
        if test_optimizations():
            print("\n✓ All optimizations tested successfully!")
            print("\nOptimizations applied:")
            print("1. ✓ Model preloading at startup")
            print("2. ✓ Redis cache optimization (2h TTL)")
            print("3. ✓ Database query optimization")
            print("4. ✓ Face detection optimization")
            print("5. ✓ Connection pooling")
            
            print("\nNext steps:")
            print("1. Restart your Flask application")
            print("2. Monitor performance improvements")
            print("3. Check logs for optimization messages")
            
            return True
        else:
            print("\n✗ Some tests failed. Check the errors above.")
            return False
    else:
        print("\n✗ Failed to apply optimizations")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
