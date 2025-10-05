"""
Konfigurasi Liveness Detection
=============================

File ini berisi konfigurasi untuk menyesuaikan sensitivitas liveness detection
sesuai dengan kebutuhan lingkungan dan kondisi pencahayaan.
"""

# Threshold untuk liveness detection
LIVENESS_THRESHOLD = 0.3  # Semakin rendah = semakin permisif (default: 0.3)

# Bobot untuk kombinasi skor
MOTION_WEIGHT = 0.4      # Bobot deteksi gerakan
BLINK_WEIGHT = 0.3       # Bobot deteksi kedipan  
TEXTURE_WEIGHT = 0.3     # Bobot analisis tekstur

# Konfigurasi Motion Detection
MOTION_SENSITIVITY = 15.0  # Semakin rendah = semakin sensitif (default: 15.0)

# Konfigurasi Texture Analysis
TEXTURE_VARIANCE_DIVISOR = 300.0  # Semakin rendah = semakin sensitif (default: 300.0)

# Konfigurasi Blink Detection
BLINK_OPENNESS_DIVISOR = 1.5  # Semakin rendah = semakin sensitif (default: 1.5)

# Debug mode
DEBUG_MODE = True  # Set True untuk melihat log debug

def get_liveness_config():
    """Mengembalikan konfigurasi liveness detection"""
    return {
        'threshold': LIVENESS_THRESHOLD,
        'motion_weight': MOTION_WEIGHT,
        'blink_weight': BLINK_WEIGHT,
        'texture_weight': TEXTURE_WEIGHT,
        'motion_sensitivity': MOTION_SENSITIVITY,
        'texture_variance_divisor': TEXTURE_VARIANCE_DIVISOR,
        'blink_openness_divisor': BLINK_OPENNESS_DIVISOR,
        'debug_mode': DEBUG_MODE
    }

def update_threshold(new_threshold: float):
    """Update threshold liveness detection"""
    global LIVENESS_THRESHOLD
    LIVENESS_THRESHOLD = new_threshold
    print(f"Liveness threshold updated to: {new_threshold}")

def update_motion_sensitivity(new_sensitivity: float):
    """Update sensitivitas motion detection"""
    global MOTION_SENSITIVITY
    MOTION_SENSITIVITY = new_sensitivity
    print(f"Motion sensitivity updated to: {new_sensitivity}")

def update_texture_sensitivity(new_divisor: float):
    """Update sensitivitas texture analysis"""
    global TEXTURE_VARIANCE_DIVISOR
    TEXTURE_VARIANCE_DIVISOR = new_divisor
    print(f"Texture sensitivity updated to: {new_divisor}")

if __name__ == "__main__":
    print("Liveness Detection Configuration")
    print("================================")
    config = get_liveness_config()
    for key, value in config.items():
        print(f"{key}: {value}")
