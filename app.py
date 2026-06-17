import cv2
import numpy as np
import os
import time
from flask import Flask, request, Response, jsonify, render_template_string
from werkzeug.utils import secure_filename
from PIL import Image
from pipeline import AttentionAnalysisPipeline

# Create upload directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the modular computer vision pipeline (SavedModel + YuNet ONNX)
pipeline = AttentionAnalysisPipeline(
    detector_path="face_detection_yunet_2023mar.onnx",
    classifier_path="focus_saved_model",
    architecture="custom" # "custom" loads the trained SavedModel; "mobilenetv2" loads the alternate arch
)

# Global flag to track streaming source
# 'webcam' or 'video' or None
current_stream_source = None
video_file_path = None

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
<title>Student Attention Analysis System</title>
<style>
body {
    background: #090c15;
    color: #e0e6ed;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
}
header {
    background: #0f1424;
    border-bottom: 2px solid #00f0ff;
    padding: 20px;
    text-align: center;
}
h1 {
    color: #00f0ff;
    margin: 0;
    font-size: 28px;
    letter-spacing: 1px;
}
.container {
    display: flex;
    flex-wrap: wrap;
    max-width: 1300px;
    margin: 30px auto;
    gap: 20px;
    padding: 0 15px;
}
.panel {
    background: #12182c;
    border-radius: 10px;
    border: 1px solid #1f2a4a;
    padding: 20px;
}
.left-panel {
    flex: 2;
    min-width: 600px;
}
.right-panel {
    flex: 1;
    min-width: 350px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.controls {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
button, label.file-label {
    background: #00f0ff;
    color: #090c15;
    border: none;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.3s;
}
button:hover, label.file-label:hover {
    background: #00b8cc;
}
input[type="file"] {
    display: none;
}
.video-display {
    width: 100%;
    min-height: 400px;
    background: #06080e;
    border-radius: 8px;
    border: 1px dashed #3a4a75;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.video-display img {
    max-width: 100%;
    max-height: 500px;
}
.stat-card {
    background: #1b2440;
    border-radius: 8px;
    padding: 15px;
    border-left: 4px solid #00f0ff;
}
.stat-title {
    font-size: 12px;
    text-transform: uppercase;
    color: #8c9bb0;
    margin-bottom: 5px;
}
.stat-value {
    font-size: 24px;
    font-weight: bold;
    color: #ffffff;
}
.analytics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
}
.console-box {
    background: #06080e;
    border: 1px solid #1f2a4a;
    border-radius: 5px;
    padding: 10px;
    font-family: monospace;
    font-size: 12px;
    height: 120px;
    overflow-y: auto;
    color: #00f0ff;
}
</style>
</head>
<body>

<header>
    <h1> Computer Vision-Based Student Attention Analysis System</h1>
</header>

<div class="container">
    <!-- Viewfinder Panel -->
    <div class="panel left-panel">
        <div class="controls">
            <form action="/upload_image" method="POST" enctype="multipart/form-data" style="display:inline;">
                <label class="file-label" for="image-upload">Upload Image</label>
                <input id="image-upload" type="file" name="image" onchange="this.form.submit()">
            </form>
            
            <form action="/upload_video" method="POST" enctype="multipart/form-data" style="display:inline;">
                <label class="file-label" for="video-upload">Upload Video File</label>
                <input id="video-upload" type="file" name="video" onchange="this.form.submit()">
            </form>

            <button onclick="location.href='/start_webcam'">Start Webcam Stream</button>
            <button onclick="location.href='/stop_stream'" style="background:#ff4d4d; color:#fff;">Stop Stream</button>
        </div>

        <div class="video-display">
            {% if img_path %}
                <img src="{{ img_path }}?t={{ timestamp }}">
            {% elif streaming %}
                <img src="/video_feed">
            {% else %}
                <div style="text-align:center; color:#5c6b8c;">
                    <p>No active feed. Upload an image, video, or start the webcam to begin analysis.</p>
                </div>
            {% endif %}
        </div>
    </div>

    <!-- Analytics Panel -->
    <div class="panel right-panel">
        <h2 style="color: #00f0ff; margin-top: 0; font-size: 20px;"> Attentiveness Analytics</h2>
        
        <div class="analytics-grid">
            <div class="stat-card">
                <div class="stat-title">Avg Attention Score</div>
                <div class="stat-value" id="val-avg-score">0.0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Active Time</div>
                <div class="stat-value" id="val-time">0s</div>
            </div>
            <div class="stat-card" style="border-left-color: #00ff66;">
                <div class="stat-title">Focused Frames</div>
                <div class="stat-value" id="val-focused">0.0%</div>
            </div>
            <div class="stat-card" style="border-left-color: #ff3333;">
                <div class="stat-title">Distracted Frames</div>
                <div class="stat-value" id="val-distracted">0.0%</div>
            </div>
        </div>

        <div class="stat-card" style="border-left-color: #ffd700;">
            <div class="stat-title">Total Faces Tracked</div>
            <div class="stat-value" id="val-faces">0</div>
        </div>

        <h3 style="color: #8c9bb0; font-size: 14px; margin-bottom: 5px;">Execution Terminal</h3>
        <div class="console-box" id="console">
            [SYS] System initialized.<br>
            [CV] Loaded YuNet ONNX Face Detection Model.<br>
            [CV] Loaded Attention Keras SavedModel.
        </div>
    </div>
</div>

<script>
    // Poll the analytics endpoint every second to update UI
    function updateAnalytics() {
        fetch('/analytics_summary')
            .then(response => response.json())
            .then(data => {
                document.getElementById('val-avg-score').innerText = data.average_attention_score + '%';
                document.getElementById('val-time').innerText = data.session_duration_sec + 's';
                document.getElementById('val-focused').innerText = data.focused_percentage + '%';
                document.getElementById('val-distracted').innerText = data.distracted_percentage + '%';
                document.getElementById('val-faces').innerText = data.total_faces_analyzed;
                
                const consoleBox = document.getElementById('console');
                const logMsg = `[ANALYSIS] Frame: ${data.total_frames_processed} | Avg Attention: ${data.average_attention_score}%`;
                
                // Only print if there are active frames
                if (data.total_frames_processed > 0) {
                    consoleBox.innerHTML += '<br>' + logMsg;
                    consoleBox.scrollTop = consoleBox.scrollHeight;
                }
            })
            .catch(err => console.log('Error fetching analytics:', err));
    }

    setInterval(updateAnalytics, 1000);
</script>

</body>
</html>
"""

@app.route("/")
def index():
    global current_stream_source
    current_stream_source = None
    return render_template_string(HTML_DASHBOARD, img_path=None, streaming=False, timestamp=int(time.time()))

@app.route("/upload_image", methods=["POST"])
def upload_image():
    global current_stream_source
    current_stream_source = None
    
    if "image" not in request.files:
        return render_template_string(HTML_DASHBOARD, img_path=None, streaming=False, timestamp=int(time.time()))
        
    file = request.files["image"]
    if file.filename == "":
        return render_template_string(HTML_DASHBOARD, img_path=None, streaming=False, timestamp=int(time.time()))

    image = Image.open(file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    pipeline.analytics.reset()
    annotated_frame_bgr, _ = pipeline.process_frame(img_bgr)

    output_path = "static/output.png"
    cv2.imwrite(output_path, annotated_frame_bgr)

    return render_template_string(HTML_DASHBOARD, img_path=output_path, streaming=False, timestamp=int(time.time()))

@app.route("/upload_video", methods=["POST"])
def upload_video():
    global current_stream_source, video_file_path
    if "video" not in request.files:
        return render_template_string(HTML_DASHBOARD, img_path=None, streaming=False, timestamp=int(time.time()))
        
    file = request.files["video"]
    if file.filename == "":
        return render_template_string(HTML_DASHBOARD, img_path=None, streaming=False, timestamp=int(time.time()))

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    
    video_file_path = path
    current_stream_source = "video"
    pipeline.analytics.reset()
    
    return render_template_string(HTML_DASHBOARD, img_path=None, streaming=True, timestamp=int(time.time()))

@app.route("/start_webcam")
def start_webcam():
    global current_stream_source
    current_stream_source = "webcam"
    pipeline.analytics.reset()
    return render_template_string(HTML_DASHBOARD, img_path=None, streaming=True, timestamp=int(time.time()))

@app.route("/stop_stream")
def stop_stream():
    global current_stream_source
    current_stream_source = None
    return render_template_string(HTML_DASHBOARD, img_path=None, streaming=False, timestamp=int(time.time()))

# Streaming Generators
def generate_webcam():
    global current_stream_source
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    failed_frames = 0
    while current_stream_source == "webcam":
        ret, frame = cap.read()
        if not ret:
            failed_frames += 1
            if failed_frames > 30:  # Break if it consistently fails (e.g. webcam unplugged)
                break
            time.sleep(0.01)
            continue
            
        failed_frames = 0
        try:
            processed, _ = pipeline.process_frame(frame)
            ret, jpeg = cv2.imencode('.jpg', processed)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"[WARN] Failed to process webcam frame: {e}")
               
    cap.release()

def generate_video():
    global current_stream_source, video_file_path
    if not video_file_path or not os.path.exists(video_file_path):
        return

    cap = cv2.VideoCapture(video_file_path)
    
    while current_stream_source == "video" and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            processed, _ = pipeline.process_frame(frame)
            ret, jpeg = cv2.imencode('.jpg', processed)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"[WARN] Failed to process video frame: {e}")
        
        # Frame-rate pacing delay (approximating 30 FPS processing output speed)
        time.sleep(0.033)
               
    cap.release()

@app.route("/video_feed")
def video_feed():
    global current_stream_source
    if current_stream_source == "webcam":
        return Response(generate_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif current_stream_source == "video":
        return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(status=204)

@app.route("/analytics_summary")
def analytics_summary():
    return jsonify(pipeline.analytics.get_summary())

if __name__ == "__main__":
    app.run(debug=False)
