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

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        image_np = np.array(image)

        (startX, startY, endX, endY) = box
        roi = image_np[startY:endY, startX:endX]

        predictor.set_image(roi)
        masks, _, _ = predictor.predict(
            box=np.array([0, 0, roi.shape[1], roi.shape[0]]),
            multimask_output=False,
        )
        mask = masks[0]

        mask = mask.astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (5, 5), 1)
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)

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
        global_image = np.array(image)
        full_img = Image.fromarray(global_image)
        full_buffered = io.BytesIO()
        full_img.save(full_buffered, format="PNG")
        full_img_str = base64.b64encode(full_buffered.getvalue()).decode("utf-8")

        global_box = [startX, startY, endX, endY]
        full_width, full_height = global_image.shape[1], global_image.shape[0]

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
    try:
        data = request.json
        full_image_data = data.get('image')
        extracted_image_data = data.get("extractedImage")
        if not full_image_data:
            return jsonify({"error": "No full image data provided"}), 400
        if not extracted_image_data:
            return jsonify({"error": "No extracted image data provided"}), 400
        points = np.array(data['points'])
        labels = np.array(data['labels'])
        box = data['box']

        full_image = Image.open(io.BytesIO(base64.b64decode(full_image_data))).convert("RGB")
        full_image_np = np.array(full_image)

        extracted_image = Image.open(io.BytesIO(base64.b64decode(extracted_image_data))).convert("RGBA")
        extracted_image_np = np.array(extracted_image)
        initial_mask = extracted_image_np[:, :, 3]  # Alpha channel from extracted image

        startX, startY, endX, endY = box
        roi = full_image_np[startY:endY, startX:endX]
        
        # Resize initial_mask to match ROI dimensions
        if initial_mask.shape != (roi.shape[0], roi.shape[1]):
            initial_mask = cv2.resize(initial_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

        predictor.set_image(full_image_np)

        # Adjust points relative to full image
        adjusted_points = points.copy()
        adjusted_points[:, 0] += startX
        adjusted_points[:, 1] += startY

        masks, _, _ = predictor.predict(
            point_coords=adjusted_points,
            point_labels=labels,
            multimask_output=True
        )

        # Crop the mask to the ROI size directly
        best_mask = masks[0]  # Select the best mask
        mask = best_mask[startY:endY, startX:endX]  # Crop to ROI
        mask = mask.astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (5, 5), 1)

        # Combine initial mask with refined mask (both should now match ROI size)
        combined_mask = np.logical_or(initial_mask, mask).astype(np.uint8) * 255

        # Apply the combined mask to the ROI
        extracted = cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
        extracted[:, :, 3] = combined_mask  # Use combined_mask directly

        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(extracted_rgb)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

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