# FTL Face Gate - InsightFace Face Recognition System

A Flask-based face recognition system using InsightFace for automatic face detection and recognition with bounding box visualization.

## Features

- **Automatic Face Recognition**: Real-time face detection and recognition using InsightFace
- **Bounding Box Visualization**: Green boxes for recognized faces, orange for unknown faces
- **Database Integration**: Stores face encodings in MySQL database
- **GymMaster API Integration**: Connects with GymMaster system for member management
- **Face Registration**: Burst photo capture for creating face encodings
- **Web Interface**: Clean, responsive web interface with camera controls

## Technology Stack

- **Backend**: Flask (Python)
- **Face Recognition**: InsightFace with ONNX Runtime
- **Database**: MySQL
- **Frontend**: HTML, CSS, JavaScript
- **Computer Vision**: OpenCV
- **API Integration**: GymMaster API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adityanuriskandar17/facerecog-insightface.git
cd facerecog-insightface
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```
DB_HOST=your_mysql_host
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=your_database_name
GYM_BASE_URL=your_gymmaster_url
GYM_API_KEY=your_gymmaster_api_key
```

4. Run the application:
```bash
python app.py
```

## Usage

1. **Main Page** (`/`): Automatic face recognition with bounding boxes
2. **Registration Page** (`/retake`): Face registration and profile management
3. **API Endpoints**:
   - `/api/recognize_open_gate`: Face recognition with bounding box coordinates
   - `/api/register_face`: Register new face encodings
   - `/api/retake_login`: Login with GymMaster credentials

## API Endpoints

### Face Recognition
- **POST** `/api/recognize_open_gate`
  - Body: `{"doorid": "19456", "image_b64": "base64_image"}`
  - Returns: Face recognition results with bounding box coordinates

### Face Registration
- **POST** `/api/register_face`
  - Body: `{"frames": ["base64_image1", "base64_image2", ...]}`
  - Returns: Registration status

### Authentication
- **POST** `/api/retake_login`
  - Body: `{"username": "user", "password": "pass"}`
  - Returns: Login status and profile data

## Database Schema

The system uses a `member` table with the following structure:
- `id`: Primary key
- `member_id`: GymMaster member ID
- `first_name`: Member's first name
- `last_name`: Member's last name
- `enc`: Face encoding (LONGTEXT)

## Features in Detail

### Automatic Face Recognition
- Continuous face detection every 2 seconds
- Real-time bounding box visualization
- Name display for recognized faces
- Confidence scoring

### Face Registration Process
1. User clicks "Register Face" button
2. Camera activates for registration
3. Burst photo capture (5 seconds, multiple frames)
4. Face encoding generation and averaging
5. Database storage of face encodings

### Bounding Box System
- Green boxes for recognized faces
- Orange boxes for unknown faces
- Real-time coordinate tracking
- Label display with names

## Development

The system is built with modularity in mind:
- Separate functions for face detection and recognition
- Database abstraction layer
- API endpoint organization
- Frontend-backend separation

## Requirements

- Python 3.8+
- MySQL 5.7+
- Modern web browser with camera support
- HTTPS for camera access (in production)

## License

This project is for educational and development purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please create an issue in the GitHub repository.
