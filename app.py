from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from segment_anything import sam_model_registry, SamPredictor
import os

app = Flask(__name__)
CORS(app)

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLOv3 
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# SAM model 
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Global variables to store image data for refinement
global_image = None
global_roi = None
global_box = None

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
    global global_image, global_roi, global_box
    data = request.json
    image_data = data['image']
    box = data['box']

    # Store the full image and box for refinement
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        global_image = np.array(image)
        global_box = box

        # Get full image dimensions
        full_height, full_width = global_image.shape[:2]

        (startX, startY, endX, endY) = box
        roi = global_image[startY:endY, startX:endX]
        global_roi = roi.copy()  # Store ROI for initial extraction

        predictor.set_image(roi)
        masks, _, _ = predictor.predict(
            box=np.array([0, 0, roi.shape[1], roi.shape[0]]),
            multimask_output=False,
        )
        mask = masks[0]

        # Convert mask to uint8 and smooth it
        mask = mask.astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (5, 5), 1)  # Apply Gaussian blur for smoothing
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)  # Use cubic interpolation

        # Apply mask to ROI with anti-aliased edges
        extracted = cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
        extracted[:, :, 3] = mask

        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(extracted_rgb)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        if not img_str.strip():
            return jsonify({"error": "Extracted image is blank"}), 400

        # Convert full image to base64 for display
        full_img = Image.fromarray(global_image)
        full_buffered = io.BytesIO()
        full_img.save(full_buffered, format="PNG")
        full_img_str = base64.b64encode(full_buffered.getvalue()).decode("utf-8")

        return jsonify({
            "image": img_str,  # Extracted ROI
            "full_image": full_img_str,  # Full original image
            "full_width": full_width,
            "full_height": full_height,
            "box": global_box  # [startX, startY, endX, endY]
        })
    except Exception as e:
        print(f"Error in /extract: {str(e)}")
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500

@app.route('/refine', methods=['POST'])
def refine():
    global global_image, global_roi, global_box
    # Check if global variables are set
    if global_image is None or global_box is None:
        print("Global variables not set: global_image or global_box is None")
        return jsonify({"error": "No image data available for refinement"}), 400

    try:
        data = request.json
        points = np.array(data['points'])  # Array of [x, y] coordinates
        labels = np.array(data['labels'])  # Array of 1 (include) or 0 (exclude)

        # Debug: Log the received points and labels
        print(f"Received points: {points}")
        print(f"Received labels: {labels}")

        # Use the full image for refinement
        predictor.set_image(global_image)

        # Adjust points to be relative to the full image
        (startX, startY, endX, endY) = global_box
        adjusted_points = points.copy()
        adjusted_points[:, 0] += startX  # Shift x coordinates
        adjusted_points[:, 1] += startY  # Shift y coordinates

        # Use the original box as a hint, but allow points to extend beyond it
        masks, scores, _ = predictor.predict(
            box=np.array(global_box),
            point_coords=adjusted_points,
            point_labels=labels,
            multimask_output=True,  # Try multiple masks to get the best one
        )

        # Select the mask with the highest score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]

        # Convert mask to uint8 and smooth it
        mask = mask.astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (5, 5), 1)  # Apply Gaussian blur for smoothing
        mask = cv2.resize(mask, (endX - startX, endY - startY), interpolation=cv2.INTER_CUBIC)  # Use cubic interpolation

        # Create a mask for the full image size
        full_mask = np.zeros((global_image.shape[0], global_image.shape[1]), dtype=np.uint8)
        full_mask[startY:endY, startX:endX] = mask

        # Apply the mask to the full image
        extracted = cv2.cvtColor(global_image, cv2.COLOR_RGB2BGRA)
        extracted[:, :, 3] = full_mask

        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(extracted_rgb)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        if not img_str.strip():
            return jsonify({"error": "Refined image is blank"}), 400

        return jsonify({"image": img_str})
    except Exception as e:
        print(f"Error in /refine: {str(e)}")
        return jsonify({"error": f"Refinement failed: {str(e)}"}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generator')
def generator():
    return render_template('generator.html')

@app.route('/extraction')
def extraction():
    return render_template('extraction.html')

@app.route('/creator')
def creator():
    return render_template('creator.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)