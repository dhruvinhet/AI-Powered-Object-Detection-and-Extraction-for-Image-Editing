from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from segment_anything import sam_model_registry, SamPredictor

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

    # Create blob from image and perform forward pass for detection
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    # Set detection confidence threshold
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Initialize lists to hold detection details
    boxes = []
    confidences = []
    class_ids = []

    # Loop over the detections
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

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Dictionary to store detections for user selection
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


# app.py
@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    image_data = data['image']
    box = data['box']

    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = np.array(image)

    (startX, startY, endX, endY) = box

    # Extract ROI
    roi = image[startY:endY, startX:endX]

    # Prepare the image for SAM and predict the mask
    predictor.set_image(roi)
    masks, _, _ = predictor.predict(
        box=np.array([0, 0, roi.shape[1], roi.shape[0]]),
        multimask_output=False,
    )
    mask = masks[0]


    # Ensure mask and roi are compatible types and shapes
    mask = mask.astype(np.uint8) * 255 # Important to convert to 0-255 for OpenCV
    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)  # Resizing to match


    # Create the extracted image with transparency
    extracted = cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    extracted[:, :, 3] = mask  # Apply the mask directly to the alpha channel

    # Convert back to RGBA for the frontend
    extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA)

    # Convert the extracted image to PIL format for editing
    img = Image.fromarray(extracted_rgb)

    # Encode the image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Check if the image is blank
    if not img_str.strip():
        return jsonify({"error": "Extracted image is blank"}), 400

    return jsonify({"image": img_str})
if __name__ == '__main__':
    app.run(debug=True)