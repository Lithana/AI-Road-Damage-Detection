from flask import Flask, request, render_template, send_from_directory, Response, url_for, redirect
from ultralytics import YOLO
import os
import random
import cv2
import json
import json
import uuid
from datetime import datetime
import requests
try:
    from gps_handler import get_hardware_gps
    HAS_GPS_HARDWARE = True
except ImportError:
    HAS_GPS_HARDWARE = False

app = Flask(__name__)

# --- SESSION TRACKING FOR LIVE WEBCAM ---
# Stores active inspection data: {session_id: {unique_ids: set(), start_time: datetime, ...}}
active_sessions = {}

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
HISTORY_FILE = 'history.json'

# Simulated GPS state for "Live" movement
gps_state = {"lat": 12.9716, "lng": 80.2446}

def get_moving_gps():
    # Try to get Hardware GPS first if on Raspberry Pi
    if HAS_GPS_HARDWARE:
        lat, lon = get_hardware_gps()
        if lat is not None:
            return lat, lon
            
    # Simulate slight movement (approx. vehicle speed) if no hardware fix
    gps_state["lat"] += random.uniform(-0.0001, 0.0001)
    gps_state["lng"] += random.uniform(-0.0001, 0.0001)
    return round(gps_state["lat"], 6), round(gps_state["lng"], 6)

def get_location_name(lat, lng):
    # Free OpenStreetMap Reverse Geocoding (Nominatim)
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
        headers = {'User-Agent': 'AeroVision/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        full_name = data.get('display_name', f"Region ({lat}, {lng})")
        address = data.get('address', {})
        
        # Try to get the most specific area name from the address fields
        # Ordered from most specific to more general
        specific_area = (
            address.get('suburb') or 
            address.get('neighbourhood') or 
            address.get('village') or 
            address.get('hamlet') or
            address.get('subdivision') or
            address.get('city_district') or 
            address.get('town') or
            address.get('city') or
            address.get('district')
        )
        
        if not specific_area:
            # Fallback extraction from display name (usually the first part is the most specific)
            parts = full_name.split(',')
            if len(parts) > 0:
                specific_area = parts[0].strip()

        return full_name, specific_area or "Unknown Area"
    except:
        return f"Sector {random.randint(100, 999)}, Infrastructure Zone", "Unknown Area"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")

def save_history(record):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.insert(0, record)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def generate_frames(source):
    # For Windows, CAP_DSHOW is often more reliable for webcams
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
        
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Run YOLO inference with tracking
            results = model.track(frame, persist=True, verbose=False, conf=0.3)
            annotated_frame = results[0].plot()
            
            # If there's an active session, track the IDs
            if active_sessions:
                latest_session_id = list(active_sessions.keys())[-1]
                if active_sessions[latest_session_id]['active']:
                    if results[0].boxes.id is not None:
                        ids = results[0].boxes.id.cpu().numpy().astype(int)
                        active_sessions[latest_session_id]['unique_ids'].update(ids)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        print(f"Released video source: {source}")

def analyze_video_full(video_path):
    # Using tracking to count unique potholes across the entire video
    results = model.track(source=video_path, stream=True, persist=True, conf=0.3)
    unique_ids = set()
    for result in results:
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            unique_ids.update(ids)
    return len(unique_ids)

# ---------------- PORTAL LANDING PAGE ----------------
@app.route('/')
def landing_page():
    return render_template('landing.html')

# ---------------- SCAN CONSOLE ROUTE ----------------
@app.route('/scan', methods=['GET', 'POST'])
def home():
    img_path = None
    video_url = None
    latitude = None
    longitude = None
    anomalies_count = 0

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html')
            
        f = request.files['file']
        if f.filename == '':
            return render_template('index.html')

        # Generate a unique ID for this analysis session
        session_id = str(uuid.uuid4())
        
        # Get coordinates for the report
        latitude, longitude = get_moving_gps()
        ext = os.path.splitext(f.filename)[1]
        upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}{ext}")
        f.save(upload_path)

        # IMAGE DETECTION
        if f.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            results = model(upload_path, conf=0.3)
            
            result_filename = f"res_{session_id}.jpg"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            results[0].save(filename=result_path)
            
            img_path = f"/results/{result_filename}"
            anomalies_count = len(results[0].boxes)

        # VIDEO DETECTION
        elif f.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_url = url_for('video_feed', filename=f"{session_id}{ext}")
            # Full analysis using tracking to get the exact unique count for the report
            anomalies_count = analyze_video_full(upload_path)

        # Severity Logic
        if anomalies_count == 0:
            assessment_status = "Perfect"
            severity = "Normal"
        elif anomalies_count < 10:
            assessment_status = "Detected"
            severity = "Medium"
        else:
            assessment_status = "Detected"
            severity = "High"

        # Get Location Info
        location_name, specific_area = get_location_name(latitude, longitude)

        record = {
            "id": session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": f.filename,
            "type": "Image" if f.filename.lower().endswith(('.jpg', '.jpeg', '.png')) else "Video",
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
            "specific_location": specific_area,
            "anomalies": anomalies_count,
            "severity": severity,
            "status": assessment_status,
            "result_file": img_path if img_path else None
        }
        save_history(record)
        
        # Redirect to the dedicated analysis page
        return redirect(url_for('analysis_page', analysis_id=session_id))

    return render_template('index.html')

@app.route('/analysis/<analysis_id>')
def analysis_page(analysis_id):
    records = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
                
    analysis_data = next((r for r in records if r['id'] == analysis_id), None)
    
    if not analysis_data:
        return "Analysis not found", 404
        
    return render_template('analysis.html', analysis=analysis_data)

@app.route('/telemetry')
def telemetry_page():
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except:
                history = []
    return render_template('telemetry.html', history=history)

# ---------------- VIDEO STREAM ROUTE ----------------
@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/webcam_feed')
def webcam_feed():
    camera_index = request.args.get('index', 0, type=int)
    return Response(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_inspection')
def start_inspection():
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        'active': True,
        'unique_ids': set(),
        'start_time': datetime.now()
    }
    return {"status": "success", "session_id": session_id}

@app.route('/stop_inspection/<session_id>')
def stop_inspection(session_id):
    if session_id not in active_sessions:
        return {"status": "error", "message": "Session not found"}, 404
        
    session = active_sessions[session_id]
    session['active'] = False
    
    anomalies_count = len(session['unique_ids'])
    
    # Severity Logic
    if anomalies_count == 0:
        assessment_status = "Perfect"
        severity = "Normal"
    elif anomalies_count < 10:
        assessment_status = "Detected"
        severity = "Medium"
    else:
        assessment_status = "Detected"
        severity = "High"
        
    # Get the latest simulated GPS location for the report
    lat, lng = get_moving_gps()
    location_name, specific_area = get_location_name(lat, lng)
        
    # Create History Record
    record = {
        "id": session_id,
        "timestamp": session['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
        "filename": "Live Webcam Stream",
        "type": "Live Stream",
        "latitude": lat,
        "longitude": lng,
        "location_name": location_name,
        "specific_location": specific_area,
        "anomalies": anomalies_count,
        "severity": severity,
        "status": assessment_status,
        "result_file": None
    }
    save_history(record)
    
    return {"status": "success", "report_id": session_id}

# ---------------- SERVE FILES ----------------
@app.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/uploads/<path:filename>')
def get_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Catch-all for any other files in root (backwards compatibility)
@app.route('/<path:filename>')
def get_file(filename):
    return send_from_directory('.', filename)

# ---------------- HISTORY ROUTES ----------------
@app.route('/history')
def history():
    records = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    
    # Calculate Frequent Location Analytics
    location_stats = {}
    total_global_anomalies = 0
    
    unique_locations = set()
    
    for record in records:
        # Backward compatibility for specific_location
        if 'specific_location' not in record:
            loc = record.get('location_name', 'Unknown Region')
            record['specific_location'] = loc.split(',')[0].strip() if loc else "Unknown"
            
        loc_short = record['specific_location']
        unique_locations.add(loc_short)
        
        count = record.get('anomalies', 0)
        
        if isinstance(count, (int, float)):
            location_stats[loc_short] = location_stats.get(loc_short, 0) + count
            total_global_anomalies += count

    frequent_location = "None"
    frequent_percentage = 0
    
    if total_global_anomalies > 0 and location_stats:
        frequent_location = max(location_stats, key=location_stats.get)
        frequent_percentage = round((location_stats[frequent_location] / total_global_anomalies) * 100, 1)

    return render_template('history.html', 
                          records=records, 
                          frequent_location=frequent_location, 
                          frequent_percentage=frequent_percentage,
                          locations=sorted(list(unique_locations)))

@app.route('/report/<report_id>')
def report(report_id):
    records = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
                
    report_data = next((r for r in records if r['id'] == report_id), None)
    
    if not report_data:
        return "Report not found", 404
        
    return render_template('report.html', report=report_data)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)