# 🚪 Set Door ID Remotely - Guide

## Overview

Fitur **Set Door ID** memungkinkan admin untuk mengatur atau mengubah Door ID pada tablet dari jarak jauh (remote) melalui admin control panel. Ini sangat berguna ketika:

- Tablet baru connect tanpa Door ID (`Door ID: None`)
- Perlu mengubah Door ID tanpa akses fisik ke tablet
- Camera tidak start karena Door ID belum di-set
- Relocating tablet ke pintu lain

---

## 🎯 Features

### 1. **Remote Door ID Configuration**
- Set Door ID dari admin panel
- No need physical access ke tablet
- Instant apply dengan auto-reload

### 2. **Real-time Update**
- Device otomatis reload dengan Door ID baru
- Camera langsung start after reload
- Status update langsung di admin panel

### 3. **Error Handling**
- Validation untuk Door ID (must be number)
- Check device connection status
- Clear error messages

---

## 🚀 How to Use

### **Step-by-Step:**

#### **1. Open Admin Control Panel**
```
http://localhost:8080/admin/control
atau
http://your-server-ip:8080/admin/control
```

#### **2. Login**
```
Username: admin
Password: admin123
```

#### **3. Locate the Device**
Device list akan show:
```
┌─────────────────────────────────────┐
│ 📱 Tablet Device - 192.168.1.100   │
│                                     │
│ 📍 Location: 192.168.1.100         │
│ 🚪 Door ID: None  ← Need to set!   │
│                                     │
│ [🗺️ Map] [🚪 Set Door ID] [🔄 Refresh] │
└─────────────────────────────────────┘
```

#### **4. Click "Set Door ID" Button**
```
Button appearance:
🚪 Set Door ID (pink/magenta gradient)
```

#### **5. Enter Door ID**
```
Popup akan muncul:
┌────────────────────────────────────┐
│ Set Door ID for Tablet Device:    │
│                                    │
│ [  1  ]  ← Enter number here      │
│                                    │
│     [OK]      [Cancel]             │
└────────────────────────────────────┘
```

#### **6. Confirm**
```
Click OK:
→ Admin panel shows: ✅ Door ID 1 set for Tablet Device. Device will reload.
→ Tablet shows: 🚪 Door ID set to 1. Reloading page...
→ After 2 seconds: Tablet auto-reload dengan ?doorid=1
→ Page reloads with Door ID
→ ✅ Success notification: "Door ID 1 - Camera Ready!"
→ Door ID badge turns GREEN
→ "Start Camera" button becomes ENABLED (clickable)
→ Click "Start Camera" → Camera starts! 📹
```

---

## 📱 What Happens on the Tablet?

### **Before Set Door ID:**
```
Browser URL: http://192.168.1.100:8080/
Status: Door ID: None (RED badge)
Start Camera Button: ❌ DISABLED (greyed out, cursor: not-allowed)
Camera: ❌ NOT STARTED (waiting for Door ID)
Tooltip: "Door ID required. Please set Door ID from admin panel."
```

### **After Admin Sets Door ID to 1:**

**Step 1: Notification appears**
```
┌────────────────────────────────────┐
│ 🚪 Door ID set to 1. Reloading... │
└────────────────────────────────────┘
(Blue notification at top of screen)
```

**Step 2: Auto-reload (2 seconds later)**
```
Browser URL: http://192.168.1.100:8080/?doorid=1
Status: Door ID: 1 (GREEN badge)
Start Camera Button: ✅ ENABLED (clickable, cursor: pointer, full opacity)
Tooltip: "Click to start camera"

Console logs:
🚪 Door ID from backend: 1
🔑 Door Token available: Yes (...)
✅ Door ID and token available, enabling Start Camera button
```

**Step 3: Success notification**
```
┌────────────────────────────────────┐
│ ✅ Door ID 1 - Camera Ready!      │
└────────────────────────────────────┘
(Green notification at top, auto-dismiss after 4 seconds)
```

**Step 4: Start Camera**
```
User clicks "Start Camera" button
→ Camera initializes
→ Video stream starts
→ Face recognition active
→ ✅ READY!
```

---

## 🔧 Technical Details

### **API Endpoint**
```python
POST /api/admin/set_door_id/<client_id>

Request Body:
{
  "door_id": 1
}

Response (Success):
{
  "ok": true,
  "message": "Door ID 1 set successfully. Device will reload."
}

Response (Error):
{
  "ok": false,
  "error": "Device not found. Device may have disconnected."
}
```

### **WebSocket Event**
```javascript
// Server emits to specific device:
socket.emit('set_door_id', {
  door_id: 1,
  timestamp: "2024-01-15T10:30:00Z",
  message: "Door ID set to 1. Reloading page..."
}, room=client_id);

// Client receives and handles:
socket.on('set_door_id', function(data) {
  // Show notification
  // Wait 2 seconds
  // Reload with ?doorid=X parameter
});
```

### **URL Parameter Magic**
```javascript
// Current URL without doorid
http://192.168.1.100:8080/

// JavaScript adds doorid parameter:
const currentUrl = new URL(window.location.href);
currentUrl.searchParams.set('doorid', doorId);

// New URL:
http://192.168.1.100:8080/?doorid=1

// Reload to this URL:
window.location.href = currentUrl.toString();
```

### **Flask Backend Reads Door ID**
```python
# In app.py
doorid = request.args.get('doorid')

# Render to HTML template
return render_template_string(INDEX_HTML, doorid=doorid)

# JavaScript can access:
const doorId = '{{ doorid }}';  // "1"
```

---

## 🐛 Troubleshooting

### **Problem 1: "Device not found" Error**

**Symptoms:**
```
❌ Failed: Device not found. Device may have disconnected. Please refresh the page and try again.
```

**Cause:**
- Device disconnected
- Device reconnected with new ID
- Admin panel showing old data

**Solution:**
```
1. Click "🔄 Reload List" button in admin panel
2. Wait for device list to refresh
3. Try "Set Door ID" again
4. ✅ Should work now!
```

**Alternative (if still fails):**
```
Open tablet browser directly with doorid:
http://192.168.1.100:8080/?doorid=1
```

### **Problem 2: Tablet Not Reloading**

**Symptoms:**
- Admin panel shows success
- But tablet still shows "Door ID: None"

**Check:**
```
1. Tablet Browser Console (F12):
   Look for: 🚪 Door ID command received: {...}
   
2. Tablet Network (F12 → Network tab):
   Look for: WebSocket connection active?

3. Server Terminal:
   Look for: ✅ Door ID X set for device Y
```

**Solution:**
```
If WebSocket disconnected:
1. Refresh tablet browser (F5)
2. Wait for reconnect
3. Try Set Door ID again from admin

If still not working:
Manual method: Add ?doorid=1 to URL manually
```

### **Problem 3: Camera Still Not Starting**

**Symptoms:**
- Door ID set successfully
- Tablet reloaded
- But camera still not starting

**Check:**
```
1. Verify Door ID in database:
   SELECT * FROM doors WHERE id = 1;
   → Should exist!

2. Check browser console:
   Look for camera errors

3. Verify URL has doorid:
   http://...?doorid=1
   ↑ Must be present!
```

**Solution:**
```
1. If door doesn't exist in database:
   → Add it manually or via admin
   
2. If browser blocks camera:
   → Grant camera permission
   
3. If URL missing doorid:
   → Set Door ID again from admin
```

### **Problem 4: Invalid Door ID Error**

**Symptoms:**
```
⚠️ Please enter a valid Door ID (number)
```

**Cause:**
- Entered non-number value
- Entered empty value
- Special characters

**Solution:**
```
✅ Valid Door IDs:
- 1
- 2
- 100
- 999

❌ Invalid Door IDs:
- abc (text)
- 1.5 (decimal - but will work, converted to 1)
- -1 (negative - but will work)
- "" (empty)
```

---

## 💡 Best Practices

### **For Initial Setup:**

1. **Prepare Door IDs**
   ```
   Plan your door numbering:
   - Lobby: Door ID 1
   - Office A: Door ID 2
   - Office B: Door ID 3
   etc.
   ```

2. **Label Tablets**
   ```
   Physical label on each tablet:
   "Tablet A - Door ID: 1"
   "Tablet B - Door ID: 2"
   ```

3. **Set Door ID Immediately**
   ```
   When new tablet connects:
   1. Note device name/IP
   2. Set correct Door ID
   3. Verify camera starts
   4. ✅ Done!
   ```

### **For Changing Door ID:**

1. **Document Changes**
   ```
   Log:
   - Old Door ID: 1
   - New Door ID: 3
   - Reason: Moved tablet
   - Date: 2024-01-15
   ```

2. **Test After Change**
   ```
   1. Verify tablet reloaded
   2. Check camera started
   3. Test face recognition
   4. Confirm logs show correct door
   ```

### **For Troubleshooting:**

1. **Check Connection First**
   ```
   Before setting Door ID:
   1. Verify device in list (green status)
   2. Check last heartbeat (recent)
   3. Then set Door ID
   ```

2. **Use Browser Console**
   ```
   F12 on tablet → Console tab
   Look for:
   ✅ GPS location obtained
   ✅ WebSocket connected
   🚪 Door ID command received
   🔄 Reloading with Door ID: X
   ```

3. **Monitor Server Logs**
   ```
   Terminal output:
   📝 Device registered: Tablet Device
   🔍 Set Door ID request for client: abc123
   ✅ Door ID 1 set for device abc123
   ```

---

## 🔐 Security

### **Authentication Required**
```
Admin panel requires login:
- Username: admin
- Password: admin123 (change in production!)

Unauthorized requests → 401 error
```

### **Device ID Validation**
```
- Check device exists in connected_devices
- Verify WebSocket connection active
- Log all Door ID changes
```

### **Input Validation**
```
- Door ID must be number
- No SQL injection (using parameterized queries)
- No XSS (input sanitized)
```

---

## 📊 Monitoring

### **Admin Panel Shows:**
```
For each device:
- Current Door ID (None, 1, 2, etc.)
- Last update timestamp
- Connection status
- GPS location (if available)
```

### **Server Logs Show:**
```
📝 Device registered: Tablet Device at 192.168.1.100
🔍 Set Door ID request for client: abc123
🔍 Connected devices: ['abc123', 'def456']
✅ Door ID 1 set for device abc123
```

### **Client Logs Show:**
```
(Browser Console - F12)
🚪 Door ID command received: {door_id: 1, message: "..."}
🔄 Reloading with Door ID: 1
✅ WebSocket connected!
📝 Device registered: Tablet Device
```

---

## 🎓 Advanced Usage

### **Bulk Door ID Assignment**

For multiple tablets at once, you can script it:

```javascript
// In browser console on admin panel:
async function setBulkDoorIds(assignments) {
  for (const [deviceId, doorId] of Object.entries(assignments)) {
    await setDoorId(deviceId, `Device ${doorId}`, doorId);
    await new Promise(r => setTimeout(r, 3000)); // Wait 3 seconds between
  }
}

// Usage:
setBulkDoorIds({
  'abc123': 1,
  'def456': 2,
  'ghi789': 3
});
```

### **Auto-Set Door ID on First Connect**

Modify registration logic to auto-assign next available Door ID:

```python
# In handle_register_device function:
if not data.get('doorid'):
    # Auto-assign next available Door ID
    existing_doorids = [d.get('doorid') for d in connected_devices.values() if d.get('doorid')]
    next_doorid = max(existing_doorids, default=0) + 1
    
    connected_devices[client_id]['doorid'] = next_doorid
    
    # Send to device
    socketio.emit('set_door_id', {
        'door_id': next_doorid,
        'message': f'Auto-assigned Door ID {next_doorid}'
    }, room=client_id)
```

---

## 📈 Statistics

### **Track Door ID Changes:**

```python
# Add logging:
door_id_changes = []

@app.route("/api/admin/set_door_id/<client_id>", methods=["POST"])
def api_set_door_id(client_id):
    # ... existing code ...
    
    # Log change:
    door_id_changes.append({
        'client_id': client_id,
        'old_doorid': connected_devices[client_id].get('doorid'),
        'new_doorid': door_id,
        'timestamp': datetime.now().isoformat(),
        'admin_ip': request.remote_addr
    })
    
    # ... rest of code ...
```

### **View Change History:**

```python
@app.route("/api/admin/door_id_history", methods=["GET"])
def api_door_id_history():
    return jsonify({
        "ok": True,
        "changes": door_id_changes[-100:]  # Last 100 changes
    })
```

---

## 📞 Support

**Still having issues?**

1. **Check all 3 log sources:**
   - Admin Panel (browser console F12)
   - Tablet (browser console F12)
   - Server Terminal

2. **Try manual URL method:**
   ```
   http://tablet-ip:8080/?doorid=1
   ```

3. **Restart if needed:**
   ```bash
   # Stop server: Ctrl+C
   # Restart:
   python app.py
   ```

4. **Verify database:**
   ```sql
   SELECT * FROM doors WHERE id = 1;
   ```

---

**Set Door ID Feature is Ready! 🎉**

Now you can configure tablets remotely without physical access!

