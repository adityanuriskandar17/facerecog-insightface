"""
Configuration untuk optimasi performa Face Recognition
====================================================

File ini berisi konfigurasi optimasi untuk:
1. Redis cache settings
2. Database optimization
3. Model configuration
4. Performance monitoring
"""

import os
from typing import Dict, Any

# Redis Optimization Settings
REDIS_OPTIMIZATION = {
    "host": os.getenv("REDIS_HOST", "127.0.0.1"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
    "password": os.getenv("REDIS_PASSWORD", None),
    "decode_responses": False,
    "socket_connect_timeout": 1,
    "socket_timeout": 1,
    "max_connections": 10,
    "retry_on_timeout": True,
    "health_check_interval": 30
}

# Cache TTL Settings (dalam detik)
CACHE_SETTINGS = {
    "member_encodings_ttl": 7200,  # 2 jam (dari 1 jam)
    "profile_cache_ttl": 3600,     # 1 jam
    "model_cache_ttl": 86400,      # 24 jam
    "refresh_interval": 3600,      # 1 jam (dari 30 menit)
    "preload_enabled": True,
    "compression_enabled": True
}

# Model Optimization Settings
MODEL_SETTINGS = {
    "name": "buffalo_s",  # Model lebih kecil dan cepat
    "providers": ["CPUExecutionProvider"],
    "det_size": (320, 320),  # Resolusi lebih kecil
    "preload_at_startup": True,
    "max_faces": 5,  # Maksimal 5 wajah per deteksi
    "min_face_size": 50  # Ukuran minimum wajah (pixel)
}

# Database Optimization Settings
DATABASE_OPTIMIZATION = {
    "connection_pool_size": 10,
    "connection_timeout": 5,
    "query_timeout": 30,
    "enable_indexes": True,
    "batch_size": 100,
    "enable_query_cache": True
}

# Performance Monitoring
PERFORMANCE_MONITORING = {
    "enable_benchmarking": True,
    "log_slow_queries": True,
    "slow_query_threshold": 1.0,  # detik
    "enable_profiling": False,
    "memory_monitoring": True
}

# Face Detection Optimization
FACE_DETECTION_OPTIMIZATION = {
    "max_image_width": 640,
    "max_image_height": 480,
    "resize_algorithm": "INTER_LINEAR",
    "enable_face_tracking": True,
    "face_tracking_threshold": 0.7,
    "enable_face_quality_check": True,
    "min_face_quality": 0.5
}

# Redis Key Patterns untuk monitoring
REDIS_KEY_PATTERNS = {
    "member_cache": "face_gate:member_encodings",
    "profile_cache": "face_gate:profile:{}",
    "model_cache": "face_gate:model:{}",
    "session_cache": "face_gate:session:{}",
    "performance_metrics": "face_gate:metrics:{}"
}

# Database Indexes yang diperlukan
REQUIRED_INDEXES = [
    "ALTER TABLE member ADD INDEX idx_enc (enc(100))",
    "ALTER TABLE member ADD INDEX idx_member_id (member_id)",
    "ALTER TABLE member ADD INDEX idx_enc_length (LENGTH(enc))",
    "ALTER TABLE member ADD INDEX idx_enc_not_null (enc) WHERE enc IS NOT NULL"
]

# Optimized Query Templates
OPTIMIZED_QUERIES = {
    "fetch_members": """
        SELECT id AS member_pk, member_id AS gym_member_id, 
               CONCAT(first_name, ' ', last_name) AS email, enc
        FROM member
        WHERE enc IS NOT NULL 
          AND enc != '' 
          AND LENGTH(enc) > 100
          AND member_id IS NOT NULL
        ORDER BY id
        LIMIT 1000
    """,
    
    "count_members": """
        SELECT COUNT(*) as total_members
        FROM member
        WHERE enc IS NOT NULL AND enc != ''
    """,
    
    "get_member_by_id": """
        SELECT id, member_id, first_name, last_name, enc
        FROM member
        WHERE member_id = %s
        LIMIT 1
    """
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "max_model_load_time": 5.0,      # detik
    "max_face_detection_time": 2.0,  # detik
    "max_database_query_time": 1.0,  # detik
    "max_redis_operation_time": 0.5, # detik
    "max_total_recognition_time": 3.0 # detik
}

# Error Handling Configuration
ERROR_HANDLING = {
    "max_retries": 3,
    "retry_delay": 1.0,  # detik
    "fallback_to_database": True,
    "fallback_to_memory_cache": True,
    "log_errors": True,
    "alert_on_errors": False
}

def get_optimization_config() -> Dict[str, Any]:
    """
    Mengembalikan konfigurasi optimasi lengkap
    """
    return {
        "redis": REDIS_OPTIMIZATION,
        "cache": CACHE_SETTINGS,
        "model": MODEL_SETTINGS,
        "database": DATABASE_OPTIMIZATION,
        "monitoring": PERFORMANCE_MONITORING,
        "face_detection": FACE_DETECTION_OPTIMIZATION,
        "redis_keys": REDIS_KEY_PATTERNS,
        "indexes": REQUIRED_INDEXES,
        "queries": OPTIMIZED_QUERIES,
        "thresholds": PERFORMANCE_THRESHOLDS,
        "error_handling": ERROR_HANDLING
    }

def validate_config() -> bool:
    """
    Validasi konfigurasi optimasi
    """
    try:
        config = get_optimization_config()
        
        # Validasi Redis settings
        assert config["redis"]["port"] > 0
        assert config["redis"]["db"] >= 0
        
        # Validasi cache settings
        assert config["cache"]["member_encodings_ttl"] > 0
        assert config["cache"]["refresh_interval"] > 0
        
        # Validasi model settings
        assert config["model"]["det_size"][0] > 0
        assert config["model"]["det_size"][1] > 0
        
        # Validasi performance thresholds
        for threshold in config["thresholds"].values():
            assert threshold > 0
        
        print("✓ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Face Recognition Optimization Configuration")
    print("==========================================")
    
    config = get_optimization_config()
    print(f"Redis Host: {config['redis']['host']}:{config['redis']['port']}")
    print(f"Cache TTL: {config['cache']['member_encodings_ttl']} seconds")
    print(f"Model: {config['model']['name']}")
    print(f"Detection Size: {config['model']['det_size']}")
    
    if validate_config():
        print("\n✓ All configurations are valid")
    else:
        print("\n✗ Configuration validation failed")
