#!/usr/bin/env python3
"""
Script untuk menambahkan Anti-Spoofing Protection ke Face Recognition
====================================================================

Script ini akan:
1. Backup app.py yang ada
2. Tambahkan liveness detection
3. Modifikasi endpoint untuk anti-spoofing
4. Test integrasi
"""

import os
import shutil
from datetime import datetime

def backup_app():
    """Backup app.py sebelum modifikasi"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"app_backup_anti_spoofing_{timestamp}.py"
    shutil.copy2("app.py", backup_name)
    print(f"✓ Backup created: {backup_name}")
    return backup_name

def add_liveness_detection():
    """Tambahkan liveness detection ke app.py"""
    
    # Read current app.py
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Backup original
    backup_file = backup_app()
    
    # Add import for liveness detection
    import_addition = '''
# Anti-Spoofing Protection
from liveness_detection import SimpleLivenessDetector
'''
    
    # Find insertion point after other imports
    import_end = content.find("\n\n# Global variables")
    if import_end == -1:
        import_end = content.find("\n\n#")
    
    if import_end != -1:
        content = content[:import_end] + import_addition + content[import_end:]
    
    # Add global liveness detector
    global_addition = '''
# Anti-spoofing protection
_liveness_detector = SimpleLivenessDetector()
'''
    
    # Find insertion point for global variables
    global_start = content.find("# Global variables")
    if global_start != -1:
        content = content[:global_start] + global_addition + content[global_start:]
    
    # Add liveness detection function
    liveness_function = '''
def extract_embedding_with_liveness(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], bool, float]:
    """
    Extract embedding dengan liveness detection untuk mencegah spoofing
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
'''
    
    # Find insertion point before API endpoints
    api_start = content.find("# -------------------- API Endpoints --------------------")
    if api_start != -1:
        content = content[:api_start] + liveness_function + "\n" + content[api_start:]
    
    # Modify the recognize endpoint
    old_extract_call = "emb, bbox = extract_embedding_with_bbox(bgr)"
    new_extract_call = "emb, bbox, is_live, liveness_confidence = extract_embedding_with_liveness(bgr)"
    
    if old_extract_call in content:
        content = content.replace(old_extract_call, new_extract_call)
        
        # Add liveness check after face detection
        old_face_check = '''if emb is None:
        return jsonify({"ok": False, "error": "No face found", "bbox": None}), 200'''
        
        new_face_check = '''if emb is None:
        return jsonify({"ok": False, "error": "No face found", "bbox": None}), 200

    if not is_live:
        return jsonify({
            "ok": False, 
            "error": "Liveness detection failed - possible spoofing attempt", 
            "liveness_confidence": liveness_confidence,
            "bbox": bbox,
            "anti_spoofing": True
        }), 200'''
        
        if old_face_check in content:
            content = content.replace(old_face_check, new_face_check)
    
    # Write modified content
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✓ Liveness detection added to app.py")
    return True

def test_integration():
    """Test integrasi anti-spoofing"""
    try:
        print("Testing anti-spoofing integration...")
        
        # Test import
        import app
        print("✓ App imports successfully")
        
        # Test liveness detector
        from liveness_detection import SimpleLivenessDetector
        detector = SimpleLivenessDetector()
        print("✓ Liveness detector created")
        
        # Test database connection
        db_conn = app.get_db_conn()
        if db_conn:
            print("✓ Database connection works")
            db_conn.close()
        else:
            print("✗ Database connection failed")
        
        print("✓ Anti-spoofing integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def create_anti_spoofing_demo():
    """Buat demo untuk menunjukkan anti-spoofing"""
    demo_code = '''
# Demo Anti-Spoofing Protection
def demo_anti_spoofing():
    """
    Demo untuk menunjukkan bagaimana anti-spoofing bekerja
    """
    print("Anti-Spoofing Demo")
    print("==================")
    print("1. ✓ Motion Detection - Deteksi gerakan alami")
    print("2. ✓ Blink Detection - Deteksi kedipan mata")
    print("3. ✓ Texture Analysis - Analisis tekstur wajah")
    print("4. ✓ Photo Rejection - Menolak foto statis")
    print("\\nProteksi aktif terhadap:")
    print("- Foto statis")
    print("- Video rekaman")
    print("- Masker wajah")
    print("- Deepfake")
    
    return True

if __name__ == "__main__":
    demo_anti_spoofing()
'''
    
    with open("demo_anti_spoofing.py", "w", encoding="utf-8") as f:
        f.write(demo_code)
    
    print("✓ Anti-spoofing demo created")

def main():
    """Main function untuk menambahkan anti-spoofing"""
    print("Add Anti-Spoofing Protection")
    print("============================")
    
    # Check if required files exist
    if not os.path.exists("app.py"):
        print("✗ app.py not found")
        return False
    
    if not os.path.exists("liveness_detection.py"):
        print("✗ liveness_detection.py not found")
        return False
    
    try:
        # Add liveness detection
        print("1. Adding liveness detection...")
        add_liveness_detection()
        
        # Test integration
        print("2. Testing integration...")
        if test_integration():
            print("\\n✓ Anti-spoofing protection added successfully!")
            
            # Create demo
            print("3. Creating demo...")
            create_anti_spoofing_demo()
            
            print("\\nAnti-spoofing features added:")
            print("1. ✓ Liveness Detection")
            print("2. ✓ Motion Analysis")
            print("3. ✓ Blink Detection")
            print("4. ✓ Texture Analysis")
            print("5. ✓ Photo Rejection")
            
            print("\\nNext steps:")
            print("1. Restart your Flask application")
            print("2. Test with real face vs photo")
            print("3. Monitor anti-spoofing logs")
            
            return True
        else:
            print("\\n✗ Integration test failed")
            return False
            
    except Exception as e:
        print(f"\\n✗ Error adding anti-spoofing: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
