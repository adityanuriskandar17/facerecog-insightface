#!/usr/bin/env python3
"""
Quick optimization script untuk memperbaiki performa face recognition
"""

import os
import shutil
from datetime import datetime

def backup_app():
    """Backup app.py"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"app_backup_{timestamp}.py"
    shutil.copy2("app.py", backup_name)
    print(f"✓ Backup created: {backup_name}")
    return backup_name

def apply_quick_optimizations():
    """Aplikasikan optimasi cepat ke app.py"""
    
    # Read current app.py
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Backup original
    backup_file = backup_app()
    
    # Optimizations to apply
    optimizations = [
        # 1. Extend Redis cache TTL
        ("REDIS_CACHE_TTL = 3600", "REDIS_CACHE_TTL = 7200  # 2 hours for better performance"),
        
        # 2. Extend cache refresh interval  
        ("_CACHE_REFRESH_INTERVAL = 1800", "_CACHE_REFRESH_INTERVAL = 3600  # 1 hour instead of 30 minutes"),
        
        # 3. Use smaller model for faster detection
        ('name="buffalo_l"', 'name="buffalo_s"  # Smaller model for faster detection'),
        
        # 4. Reduce detection size for speed
        ("det_size=(640, 640)", "det_size=(320, 320)  # Smaller detection size for speed"),
        
        # 5. Add connection pooling to Redis
        ("decode_responses=False,", "decode_responses=False,\n            max_connections=10,\n            retry_on_timeout=True,"),
    ]
    
    # Apply optimizations
    for old, new in optimizations:
        if old in content:
            content = content.replace(old, new)
            print(f"✓ Applied: {old[:30]}...")
        else:
            print(f"⚠ Not found: {old[:30]}...")
    
    # Write optimized content
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✓ Quick optimizations applied!")
    return True

def add_startup_optimization():
    """Tambahkan preload di startup"""
    
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add startup optimization after imports
    startup_code = '''
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
        redis_conn = get_redis_conn()
        if redis_conn:
            print("   ✓ Redis connected")
        else:
            print("   ✗ Redis connection failed")
    except Exception as e:
        print(f"   ✗ Redis test failed: {e}")
    
    print("DEBUG: Startup optimization completed!")

# Call startup optimization when module loads
if __name__ == "__main__":
    startup_optimization()
# ================================================================
'''
    
    # Find insertion point (after imports, before first function)
    insert_point = content.find("\n\n# Global variables")
    if insert_point == -1:
        insert_point = content.find("\n\n#")
    
    if insert_point != -1:
        new_content = content[:insert_point] + startup_code + content[insert_point:]
    else:
        new_content = content + startup_code
    
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✓ Startup optimization added!")

def main():
    """Main optimization function"""
    print("Face Recognition Quick Optimization")
    print("===================================")
    
    if not os.path.exists("app.py"):
        print("✗ app.py not found")
        return False
    
    try:
        # Apply quick optimizations
        print("1. Applying quick optimizations...")
        apply_quick_optimizations()
        
        # Add startup optimization
        print("2. Adding startup optimization...")
        add_startup_optimization()
        
        print("\n✓ All optimizations applied successfully!")
        print("\nOptimizations applied:")
        print("1. ✓ Extended Redis cache TTL to 2 hours")
        print("2. ✓ Extended cache refresh to 1 hour") 
        print("3. ✓ Changed to smaller model (buffalo_s)")
        print("4. ✓ Reduced detection size to 320x320")
        print("5. ✓ Added Redis connection pooling")
        print("6. ✓ Added startup preloading")
        
        print("\nNext steps:")
        print("1. Restart your Flask application")
        print("2. Check logs for 'Startup optimization' messages")
        print("3. Monitor face recognition speed")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
