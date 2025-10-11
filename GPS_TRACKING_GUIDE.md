# 📍 GPS Tracking Feature Guide

## Overview

Sistem GPS tracking memungkinkan admin untuk mengetahui lokasi real-time dari setiap device (tablet) yang terkoneksi ke sistem.

---

## 🎯 Features

### 1. **Automatic GPS Detection**
- Setiap device secara otomatis mengirim GPS location saat connect
- GPS update setiap 5 menit untuk real-time tracking
- Support high-accuracy GPS positioning

### 2. **GPS Display in Admin Panel**
- Menampilkan koordinat GPS (latitude, longitude)
- Menampilkan accuracy level (dalam meter)
- Link langsung ke Google Maps
- Visual indicator (green = GPS available, red = not available)

### 3. **Google Maps Integration**
- Click "Map" button untuk membuka Google Maps
- Zoom level 17 (street level detail)
- Direct navigation link

---

## 🚀 How It Works

### **Client Side (Tablet)**

1. **Browser Request GPS Permission**
   ```
   When tablet connects:
   → Browser asks: "Allow location access?"
   → User taps "Allow"
   → GPS data sent to server
   ```

2. **GPS Data Collected**
   - Latitude
   - Longitude
   - Accuracy (dalam meter)
   - Timestamp

3. **Auto-Update**
   - GPS location update setiap 5 menit
   - Tetap accurate tanpa reload page

### **Server Side**

1. **GPS Data Storage**
   ```python
   connected_devices[client_id] = {
       'gps_latitude': -6.200000,
       'gps_longitude': 106.816666,
       'gps_accuracy': 15.5,
       'gps_timestamp': 1634567890000
   }
   ```

2. **API Endpoints**
   - `/api/admin/devices` - All devices with GPS
   - `/api/admin/device_location/<id>` - Specific device GPS
   - `/api/admin/devices_with_gps` - Only devices with valid GPS

---

## 📱 Usage

### **Admin Panel View**

```
Device Card:
┌─────────────────────────────────────┐
│ 📱 Tablet Device - Cabang Jakarta  │
│                                     │
│ 📍 GPS: -6.200000, 106.816666      │
│    (±15m) [Click to open Maps]     │
│                                     │
│ [Map] [Refresh]                    │
└─────────────────────────────────────┘
```

### **GPS Status Indicators**

**✅ GPS Available (Green)**
```
📍 GPS: -6.200000, 106.816666 (±15m)
   [Clickable Google Maps link]
```

**❌ GPS Not Available (Red)**
```
⚠️ GPS: User denied Geolocation
or
⚠️ GPS: Geolocation not supported
or
⚠️ GPS: Not available
```

---

## 🔧 Setup & Configuration

### **1. Browser Permissions**

**First Time Access:**
```
Tablet akan show popup:
"Allow [your-domain] to access your location?"

→ Tap "Allow" atau "Izinkan"
```

**Permission Denied:**
```
If user tap "Block":
→ GPS will show as "User denied Geolocation"
→ To fix: Clear browser data and reload
```

### **2. GPS Settings**

**High Accuracy Mode:**
```javascript
{
  enableHighAccuracy: true,  // Use GPS hardware
  timeout: 10000,            // 10 seconds timeout
  maximumAge: 300000         // Cache 5 minutes
}
```

**To Change Settings:**
Edit `app.py` line ~5323:
```javascript
navigator.geolocation.getCurrentPosition(
  successCallback,
  errorCallback,
  {
    enableHighAccuracy: true,  // ← Change this
    timeout: 10000,            // ← Or this
    maximumAge: 300000         // ← Or this
  }
);
```

### **3. Update Frequency**

**Current: Every 5 minutes**

To change, edit `app.py` line ~5277:
```javascript
// Update GPS location every 5 minutes
if (Date.now() % 300000 < 30000) {  // ← 300000ms = 5 minutes
  getLocationAndRegister();
}
```

Example: Change to 10 minutes:
```javascript
if (Date.now() % 600000 < 30000) {  // 600000ms = 10 minutes
```

---

## 🗺️ API Documentation

### **1. Get All Devices with GPS**

```bash
GET /api/admin/devices
```

**Response:**
```json
{
  "ok": true,
  "devices": [
    {
      "id": "abc123",
      "name": "Tablet Device",
      "location": "192.168.1.100",
      "doorid": 1,
      "gps": {
        "latitude": -6.200000,
        "longitude": 106.816666,
        "accuracy": 15.5,
        "timestamp": 1634567890000,
        "available": true
      }
    }
  ],
  "total": 1
}
```

### **2. Get Specific Device Location**

```bash
GET /api/admin/device_location/<client_id>
```

**Response:**
```json
{
  "ok": true,
  "location": {
    "device_id": "abc123",
    "device_name": "Tablet Device",
    "ip": "192.168.1.100",
    "gps": {
      "latitude": -6.200000,
      "longitude": 106.816666,
      "accuracy": 15.5,
      "timestamp": 1634567890000,
      "available": true,
      "google_maps_url": "https://www.google.com/maps?q=-6.200000,106.816666"
    }
  }
}
```

### **3. Get Only Devices with Valid GPS**

```bash
GET /api/admin/devices_with_gps
```

**Response:**
```json
{
  "ok": true,
  "devices": [
    {
      "id": "abc123",
      "name": "Tablet Device",
      "doorid": 1,
      "latitude": -6.200000,
      "longitude": 106.816666,
      "accuracy": 15.5,
      "timestamp": 1634567890000
    }
  ],
  "total": 1
}
```

---

## 🐛 Troubleshooting

### **Problem 1: GPS Not Showing**

**Symptoms:**
```
⚠️ GPS: Not available
```

**Solutions:**

1. **Check Browser Permission**
   ```
   Chrome: Settings → Privacy → Site Settings → Location
   → Check if your domain is allowed
   ```

2. **Check HTTPS**
   ```
   GPS requires HTTPS (except localhost)
   If using ngrok: Automatic HTTPS ✅
   ```

3. **Check Browser Console**
   ```
   Press F12 → Console tab
   Look for:
   ✅ GPS location obtained: {lat: ..., lng: ...}
   or
   ❌ GPS location failed: [error message]
   ```

### **Problem 2: GPS Permission Prompt Not Showing**

**Cause:** Permission already blocked

**Fix:**
```
Method 1: Clear Site Data
Chrome → Settings → Privacy → Clear Browsing Data
→ Select "Cookies and site data"
→ Clear → Reload page

Method 2: Reset Permissions
Chrome → Address Bar → 🔒 icon
→ Permissions → Location → "Reset"
→ Reload page
```

### **Problem 3: GPS Inaccurate**

**Cause:** Using WiFi location instead of GPS

**Fix:**
```
1. Check tablet GPS is ON
   Settings → Location → ON

2. Check high accuracy mode
   Settings → Location → Mode → "High accuracy"

3. Use outdoor (better GPS signal)
```

### **Problem 4: GPS Not Updating**

**Check Logs:**
```bash
Server Terminal:
📍 GPS: -6.200000, 106.816666 (±15m)  # Should appear every 5 min

Browser Console:
✅ GPS location obtained: {...}       # Should appear every 5 min
```

**If not updating:**
```javascript
// Check heartbeat interval
// Should run every 30 seconds
setInterval(function() {
  socket.emit('heartbeat', ...);
  
  // GPS update every 5 minutes
  if (Date.now() % 300000 < 30000) {
    getLocationAndRegister();
  }
}, 30000);
```

---

## 📊 GPS Accuracy

### **Accuracy Levels**

| Accuracy | Description | Typical Source |
|----------|-------------|----------------|
| 0-10m | Excellent | GPS + GLONASS |
| 10-50m | Good | GPS only |
| 50-100m | Fair | WiFi + Cell Tower |
| 100m+ | Poor | Cell Tower only |

### **Improving Accuracy**

1. **Enable High Accuracy Mode** (already enabled)
2. **Use GPS hardware** (not WiFi location)
3. **Clear view of sky** (for tablets with GPS)
4. **Wait for GPS fix** (may take 30-60 seconds first time)

---

## 🔐 Privacy & Security

### **Data Storage**

- GPS data stored **in-memory only** (RAM)
- **Not saved to database**
- Data lost when server restarts
- No persistent GPS history

### **Privacy Compliance**

**User Control:**
- Browser asks permission first
- User can deny anytime
- Clear indicator when GPS is used

**Data Minimization:**
- Only collect coordinates & accuracy
- No reverse geocoding (address lookup)
- No GPS history tracking

**Transparency:**
- GPS status clearly shown
- Error messages explain why GPS failed

---

## 🎓 Best Practices

### **For Admin:**

1. **Check GPS Accuracy**
   - Look at ±Xm value
   - <50m = reliable
   - >100m = may be inaccurate

2. **Verify Location**
   - Click "Map" button to verify
   - Cross-check with known device location

3. **Monitor GPS Health**
   - Green indicators = GPS working
   - Red indicators = need attention

### **For Tablet Setup:**

1. **Grant GPS Permission**
   - Always "Allow" when prompted
   - Check permission in browser settings

2. **Enable Location Services**
   - Settings → Location → ON
   - Mode: High accuracy

3. **Test GPS**
   - Check admin panel shows green GPS
   - Click "Map" to verify location

---

## 📈 Future Enhancements

**Possible Additions:**

1. **GPS History**
   - Track device movement over time
   - Heatmap visualization

2. **Geofencing**
   - Alert if device leaves authorized area
   - Automatic actions based on location

3. **Map View**
   - Show all devices on single map
   - Real-time device positions

4. **Location Analytics**
   - Distance traveled
   - Time at location
   - Movement patterns

---

## 📞 Support

**GPS Issues?**

1. Check Browser Console (F12)
2. Check Server Terminal logs
3. Verify permissions in browser
4. Test with simple GPS app first

**Need Help?**

Check logs:
```bash
# Server logs show:
📍 GPS: -6.200000, 106.816666 (±15m)
or
⚠️ GPS: [error message]
```

---

**GPS Tracking Feature is Ready! 🎉**

Setiap device sekarang akan otomatis mengirim GPS location saat connect ke server.

