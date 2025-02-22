from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load Mask R-CNN network (for segmentation)
net = cv2.dnn.readNetFromTensorflow("mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
                                    "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image = Image.open(file.stream)
    image = np.array(image)

    (H, W) = image.shape[:2]

    # Create blob from image and perform forward pass for detection and segmentation
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])

    # Set detection confidence threshold
    conf_threshold = 0.5

    # Dictionary to store detections for user selection
    detections = []

    # Loop over the detections
    for i in range(0, boxes.shape[2]):
        score = boxes[0, 0, i, 2]
        if score > conf_threshold:
            class_id = int(boxes[0, 0, i, 1])
            if class_id - 1 < len(classes):
                label = classes[class_id - 1]  # COCO dataset: classes index starts at 1
            else:
                label = "Unknown"

            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            # Save detection details
            detections.append({
                "label": label,
                "score": float(score),
                "box": [int(startX), int(startY), int(endX), int(endY)],
                "mask": masks[i, class_id].tolist()
            })

    return jsonify(detections)

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    image_data = data['image']
    box = data['box']
    mask = np.array(data['mask'])

    # Decode the image
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = np.array(image)

    (startX, startY, endX, endY) = box

    # Extract ROI for segmentation mask processing
    roi = image[startY:endY, startX:endX]

    # The mask is output at a fixed size; resize to match ROI dimensions with high-quality interpolation
    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Apply a smoothing filter to the mask to reduce jagged edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Threshold the mask to obtain a binary mask
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    mask = mask.astype("uint8")

    # Create a high-quality extracted image with transparent background
    # First, create an empty BGRA image
    extracted = cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)

    # Set pixels outside the mask to be transparent
    extracted[mask == 0] = (0, 0, 0, 0)

    # Convert the extracted image to RGB format for frontend usage
    extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA)

    # Convert the extracted image to PIL format for editing
    img = Image.fromarray(extracted_rgb)

    # Encode the image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"image": img_str})

if __name__ == '__main__':
    app.run(debug=True)