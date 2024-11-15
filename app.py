import os
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Create a Flask app
app = Flask(__name__)

# Folder to store uploaded and processed videos
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

# Create directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Configure the app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLO model
best_model = YOLO('model/best.pt')

# Define deque for averaging damage percentage
damage_deque = deque(maxlen=20)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded video
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the video (e.g., detect road damage)
    processed_video_path = process_video(filepath, filename)

    return jsonify({"message": "Video processed successfully", "processed_video": processed_video_path})

def process_video(filepath, filename):
    video_path = filepath
    processed_video_path = os.path.join(PROCESSED_FOLDER, filename)

    # Open video for processing
    cap = cv2.VideoCapture(video_path)

    # Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and VideoWriter to save processed video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Perform inference on the frame
            results = best_model.predict(source=frame, imgsz=640, conf=0.25)
            processed_frame = results[0].plot(boxes=False)

            # Initialize percentage_damage
            percentage_damage = 0

            if results[0].masks is not None:
                total_area = 0
                masks = results[0].masks.data.cpu().numpy()
                image_area = frame.shape[0] * frame.shape[1]  # total number of pixels
                for mask in masks:
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    total_area += cv2.contourArea(contour[0])
                
                percentage_damage = (total_area / image_area) * 100

            # Update deque and calculate smoothed damage percentage
            damage_deque.append(percentage_damage)
            smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)

            # Add annotation to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_position = (40, 80)
            font_color = (255, 255, 255)  # White color
            background_color = (0, 0, 255)  # Red background
            cv2.line(processed_frame, (text_position[0], text_position[1] - 10), (text_position[0] + 350, text_position[1] - 10), background_color, 40)
            cv2.putText(processed_frame, f'Road Damage: {smoothed_percentage_damage:.2f}%', text_position, font, 1, font_color, 2, cv2.LINE_AA)

            # Save processed frame to output video
            out.write(processed_frame)
        else:
            break

    cap.release()
    out.release()

    return processed_video_path

@app.route('/processed/<filename>', methods=['GET'])
def processed_video(filename):
    video_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Video not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
