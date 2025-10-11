# 🚀 Panduan Remote Control & Auto-Refresh untuk Tablet di Cabang

## ✅ Apa yang Sudah Ditambahkan?

Sistem sekarang dilengkapi dengan **Real-time WebSocket Control** yang memungkinkan Anda mengontrol semua tablet di cabang secara remote tanpa perlu datang langsung.

---

## 📦 Instalasi

### 1. Install Dependencies Baru
```bash
pip install flask-socketio python-socketio
```

Atau install semua dependencies:
```bash
pip install -r requirements.txt
```

### 2. Jalankan Server
```bash
python app.py
```

Server akan berjalan di: `http://0.0.0.0:8080`

---

## 🎯 Fitur-Fitur Baru

### ✨ 1. **Auto-Refresh Remote**
- Refresh semua tablet sekaligus dari admin panel
- Refresh tablet tertentu saja
- Tidak perlu datang ke cabang lagi!

### 📱 2. **Device Monitoring**
- Lihat semua tablet yang sedang online
- Monitor lokasi, IP address, dan waktu koneksi
- Real-time device count

### 🔔 3. **Push Notifications**
- Kirim notifikasi ke semua devices
- Support berbagai tipe: info, success, warning, error

### 💓 4. **Connection Health**
- Auto-reconnect jika koneksi terputus
- Heartbeat monitoring setiap 30 detik
- Status koneksi real-time

---

## 🖥️ Cara Menggunakan

### Untuk Admin (Remote Control)

#### 1. Login ke Admin Panel
```
http://your-server:8080/admin/login
```
- Username: (sesuai ADMIN_USERNAME di .env)
- Password: (sesuai ADMIN_PASSWORD di .env)

#### 2. Akses Control Panel
```
http://your-server:8080/admin/control
```

#### 3. Fitur yang Tersedia:

**a. Refresh All Devices**
- Klik tombol "Refresh All Devices"
- Semua tablet akan refresh otomatis dalam 1.5 detik
- Berguna saat ada update kode atau perubahan sistem

**b. Refresh Device Tertentu**
- Pilih device dari list
- Klik tombol "Refresh" pada device tersebut
- Hanya device tersebut yang akan refresh

**c. Monitor Device Status**
- Lihat berapa tablet yang online
- Info detail: nama, lokasi, IP, door ID, waktu koneksi
- Auto-refresh list setiap 5 detik

---

### Untuk Tablet di Cabang

Tablet akan **OTOMATIS** terhubung ke server saat membuka halaman face recognition.

**Yang Terjadi Otomatis:**
1. ✅ Koneksi ke WebSocket server
2. ✅ Register device info (nama, lokasi, IP)
3. ✅ Heartbeat setiap 30 detik
4. ✅ Listen untuk perintah refresh dari admin
5. ✅ Auto-reconnect jika koneksi terputus

**Tidak perlu setting apapun di tablet!** 🎉

---

## 🧪 Testing dengan Ngrok

### 1. Jalankan Ngrok
```bash
ngrok http 8080
```

Ngrok akan memberikan URL seperti:
```
https://abc123.ngrok-free.app
```

### 2. Buka di Tablet
```
https://abc123.ngrok-free.app/?doorid=1
```

### 3. Buka Admin Panel di Laptop Anda
```
https://abc123.ngrok-free.app/admin/login
```

Login, lalu akses:
```
https://abc123.ngrok-free.app/admin/control
```

### 4. Test Refresh
- Tablet Anda akan muncul di list "Connected Devices"
- Klik "Refresh All Devices" atau "Refresh" pada device tertentu
- Tablet akan refresh otomatis!

---

## 🔧 API Endpoints (untuk Advanced Users)

### Get Connected Devices
```bash
GET /api/admin/devices
Authorization: Admin session required

Response:
{
  "ok": true,
  "devices": [
    {
      "id": "client_id_123",
      "name": "Tablet Device",
      "type": "Tablet",
      "location": "192.168.1.100",
      "doorid": "1",
      "ip": "192.168.1.100",
      "connected_at": "2024-01-01T10:00:00",
      "last_heartbeat": "2024-01-01T10:05:00"
    }
  ],
  "total": 1
}
```

### Broadcast Refresh to All Devices
```bash
POST /api/admin/broadcast_refresh
Authorization: Admin session required
Content-Type: application/json

{
  "message": "Update tersedia, refreshing..."
}

Response:
{
  "ok": true,
  "message": "Refresh command sent to 5 devices",
  "device_count": 5
}
```

### Refresh Specific Device
```bash
POST /api/admin/refresh_device
Authorization: Admin session required
Content-Type: application/json

{
  "client_id": "client_id_123",
  "message": "Refresh device khusus"
}

Response:
{
  "ok": true,
  "message": "Refresh command sent to device client_id_123"
}
```

### Send Notification to All Devices
```bash
POST /api/admin/send_notification
Authorization: Admin session required
Content-Type: application/json

{
  "notification": "Server akan maintenance dalam 5 menit",
  "type": "warning"
}

Response:
{
  "ok": true,
  "message": "Notification sent to 5 devices"
}
```

---

## 🔒 Security Notes

1. **Admin Authentication Required**: Semua endpoint admin memerlukan login
2. **CORS Enabled**: Untuk testing dengan ngrok
3. **WebSocket Security**: 
   - Connection tracking per device
   - Auto-disconnect inactive connections
   - IP address logging

---

## 🚀 Workflow Deployment

### Skenario: Update Kode dari Kantor Pusat

1. **Developer Update Kode**
   ```bash
   git pull origin main
   ```

2. **Restart Server**
   ```bash
   # Server akan restart otomatis jika menggunakan supervisor/systemd
   ```

3. **Refresh Semua Tablet dari Admin Panel**
   - Buka: `http://your-server:8080/admin/control`
   - Klik: "Refresh All Devices"
   - ✅ Semua tablet di semua cabang akan refresh otomatis!

4. **Verifikasi**
   - Check device count di admin panel
   - Pastikan semua tablet reconnect setelah refresh

---

## 🐛 Troubleshooting

### Problem: Tablet tidak muncul di device list

**Solusi:**
1. Check koneksi internet tablet
2. Pastikan tablet membuka halaman face recognition
3. Check browser console di tablet (F12) untuk error
4. Pastikan server running dan accessible

### Problem: WebSocket connection failed

**Solusi:**
1. Check firewall settings (port 8080 harus open)
2. Jika pakai nginx/reverse proxy, pastikan WebSocket support enabled:
   ```nginx
   location / {
       proxy_pass http://localhost:8080;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
   }
   ```

### Problem: Refresh tidak bekerja

**Solusi:**
1. Check browser console di tablet
2. Pastikan WebSocket connected (lihat log console)
3. Test dengan single device refresh dulu
4. Check admin session (login ulang)

---

## 📊 Monitoring & Logs

### Server Logs
```bash
# Saat tablet connect:
📱 Device connected: abc123 from 192.168.1.100
   Total connected devices: 5

# Saat tablet disconnect:
📴 Device disconnected: abc123
   Total connected devices: 4

# Saat refresh broadcast:
🔄 Broadcast refresh to 5 devices

# Saat device register:
📝 Device registered: Tablet Device at 192.168.1.100
```

### Browser Console (Tablet)
```
🔌 Initializing WebSocket connection...
✅ WebSocket connected!
✅ Device registered
💓 Heartbeat acknowledged
🔄 Received refresh command from server
```

---

## 🎉 Keuntungan Sistem Ini

1. ✅ **Hemat Waktu & Biaya**: Tidak perlu datang ke cabang untuk update
2. ✅ **Real-time**: Refresh instant ke semua devices
3. ✅ **Reliable**: Auto-reconnect & heartbeat monitoring
4. ✅ **Scalable**: Support unlimited devices
5. ✅ **Easy to Use**: Admin panel yang simple dan intuitif
6. ✅ **Compatible dengan Ngrok**: Perfect untuk testing

---

## 📞 Support

Jika ada masalah atau pertanyaan:
- Check troubleshooting section di atas
- Review server logs
- Check browser console di tablet
- Pastikan network connection stabil

---

**Happy Coding! 🚀**

*FTL IT Developer Team*

