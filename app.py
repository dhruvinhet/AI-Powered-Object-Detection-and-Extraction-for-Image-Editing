from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from segment_anything import sam_model_registry, SamPredictor
import os
from diffusers import StableDiffusionPipeline

app = Flask(__name__)
CORS(app)

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLOv3 network (for detection)
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Load SAM model (for segmentation)
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image = Image.open(file.stream)
    image = np.array(image)

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    conf_threshold = 0.5
    nms_threshold = 0.4

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))
                boxes.append([startX, startY, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            (startX, startY, width, height) = box
            endX = startX + width
            endY = startY + height
            label = classes[class_ids[i]]
            detections.append({
                "label": label,
                "score": confidences[i],
                "box": [startX, startY, endX, endY]
            })

    return jsonify(detections)

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    image_data = data['image']
    box = data['box']

    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = np.array(image)

    (startX, startY, endX, endY) = box
    roi = image[startY:endY, startX:endX]

    predictor.set_image(roi)
    masks, _, _ = predictor.predict(
        box=np.array([0, 0, roi.shape[1], roi.shape[0]]),
        multimask_output=False,
    )
    mask = masks[0]

    mask = mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    extracted = cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    extracted[:, :, 3] = mask

    extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA)
    img = Image.fromarray(extracted_rgb)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if not img_str.strip():
        return jsonify({"error": "Extracted image is blank"}), 400

    return jsonify({"image": img_str})

@app.route('/animate', methods=['POST'])
def animate():
    data = request.json
    image_data = data['image']  # Base64 encoded extracted image
    label = data['label']       # Detected object label (e.g., "car", "dog")

    # Decode the base64 image
    img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGBA")
    img = np.array(img)

    # Define video parameters
    width, height = 640, 480  # Output video resolution
    fps = 30
    duration = 3  # Seconds
    frames = fps * duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Resize extracted image to fit animation
    obj_height, obj_width = img.shape[:2]
    scale = min(height / 2 / obj_height, width / 4 / obj_width)
    new_size = (int(obj_width * scale), int(obj_height * scale))
    obj = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # Create a simple background based on object type
    def create_background(label):
        bg = np.zeros((height, width, 4), dtype=np.uint8)
        if label == "car":
            # Road-like background
            bg[:, :, 0] = 50  # Blue-ish gray (B)
            bg[:, :, 1] = 50  # (G)
            bg[:, :, 2] = 50  # (R)
            bg[height-100:, :, :3] = [100, 100, 100]  # Gray road
            bg[:, :, 3] = 255  # Fully opaque
        elif label in ["dog", "cat", "horse"]:
            # Grass-like background
            bg[:, :, 0] = 34   # (B)
            bg[:, :, 1] = 139  # Green (G)
            bg[:, :, 2] = 34   # (R)
            bg[:, :, 3] = 255  # Fully opaque
        else:
            # Sky blue default background
            bg[:, :, 0] = 135  # (B)
            bg[:, :, 1] = 206  # (G)
            bg[:, :, 2] = 235  # (R)
            bg[:, :, 3] = 255  # Fully opaque
        return bg

    # Define animation based on label
    for frame in range(frames):
        canvas = create_background(label)  # Initialize with background

        if label == "car":
            # Car "running" with slight rotation
            x_pos = int((frame / frames) * (width - new_size[0]))
            y_pos = height - new_size[1] - 50  # Near bottom (road)
            angle = np.sin(frame * 0.1) * 5  # Slight tilt
            rotated = rotate_image(obj, angle)
            place_image(canvas, rotated, x_pos, y_pos)

        elif label == "dog":
            # Dog "walking" with bounce and flip
            x_pos = int((frame / frames) * (width - new_size[0]))
            y_offset = int(10 * np.sin(frame * 0.2))  # Bounce
            y_pos = height - new_size[1] - 50 + y_offset
            if frame % 20 < 10:  # Flip every 10 frames for "walking"
                walking_obj = cv2.flip(obj, 1)  # Horizontal flip
            else:
                walking_obj = obj
            place_image(canvas, walking_obj, x_pos, y_pos)

        elif label == "cat":
            # Cat "sleeping" (slight scale change for breathing)
            x_pos = width // 2 - new_size[0] // 2  # Center horizontally
            y_pos = height - new_size[1] - 50      # Near bottom
            scale_factor = 1 + 0.05 * np.sin(frame * 0.1)  # Breathing effect
            scaled = cv2.resize(obj, (int(new_size[0] * scale_factor), int(new_size[1] * scale_factor)))
            place_image(canvas, scaled, x_pos - int(new_size[0] * (scale_factor - 1) / 2), y_pos)

        elif label == "horse":
            # Horse "galloping" with bigger bounce
            x_pos = int((frame / frames) * (width - new_size[0]))
            y_offset = int(20 * np.sin(frame * 0.3))  # Larger bounce
            y_pos = height - new_size[1] - 50 + y_offset
            place_image(canvas, obj, x_pos, y_pos)

        else:
            # Default: slide across screen with slight rotation
            x_pos = int((frame / frames) * (width - new_size[0]))
            y_pos = height // 2 - new_size[1] // 2
            angle = np.sin(frame * 0.1) * 10  # Gentle rotation
            rotated = rotate_image(obj, angle)
            place_image(canvas, rotated, x_pos, y_pos)

        # Write frame to video
        out.write(cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR))

    out.release()
    return send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name=f"{label}_animation.mp4")

# Helper function to rotate an image
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

# Corrected helper function to place an image on the canvas with transparency
def place_image(canvas, obj, x, y):
    obj_h, obj_w = obj.shape[:2]
    # Ensure placement stays within bounds
    x, y = max(0, x), max(0, y)
    if x + obj_w > canvas.shape[1] or y + obj_h > canvas.shape[0]:
        return  # Skip if out of bounds

    # Extract the region of interest (ROI) from the canvas
    roi = canvas[y:y + obj_h, x:x + obj_w, :3]  # RGB only for blending
    mask = obj[:, :, 3]  # Alpha channel of the object
    mask_inv = cv2.bitwise_not(mask)

    # Ensure ROI and mask have the same dimensions
    if roi.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_inv = cv2.bitwise_not(mask)

    # Prepare background and foreground (RGB only)
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(obj[:, :, :3], obj[:, :, :3], mask=mask)

    # Combine background and foreground
    combined = cv2.add(bg, fg)

    # Update the canvas with the combined RGB and original alpha
    canvas[y:y + obj_h, x:x + obj_w, :3] = combined
    canvas[y:y + obj_h, x:x + obj_w, 3] = cv2.bitwise_or(canvas[y:y + obj_h, x:x + obj_w, 3], mask)
    
if __name__ == '__main__':
    app.run(debug=True)