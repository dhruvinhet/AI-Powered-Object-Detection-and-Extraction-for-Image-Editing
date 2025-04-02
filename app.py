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

        # Preprocess image for better detection
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX)

        (startX, startY, endX, endY) = box
        # Shrink bounding box slightly
        padding = 0.9
        width = endX - startX
        height = endY - startY
        startX = int(startX + width * (1 - padding) / 2)
        startY = int(startY + height * (1 - padding) / 2)
        endX = int(endX - width * (1 - padding) / 2)
        endY = int(endY - height * (1 - padding) / 2)
        roi = image_np[startY:endY, startX:endX]

        predictor.set_image(roi)
        masks, scores, _ = predictor.predict(
            box=np.array([0, 0, roi.shape[1], roi.shape[0]]),
            multimask_output=True
        )

        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        mask = mask.astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (5, 5), 1)

        if mask.shape != roi.shape[:2]:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)

        extracted = cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
        extracted[:, :, 3] = mask

        full_height, full_width = image_np.shape[:2]
        full_extracted = np.zeros((full_height, full_width, 4), dtype=np.uint8)
        full_extracted[startY:endY, startX:endX] = extracted

        extracted_rgb = cv2.cvtColor(full_extracted, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(extracted_rgb)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        full_img = Image.fromarray(image_np)
        full_buffered = io.BytesIO()
        full_img.save(full_buffered, format="PNG")
        full_img_str = base64.b64encode(full_buffered.getvalue()).decode("utf-8")

        # Log the dimensions for debugging
        print(f"Original image dimensions: width={image_np.shape[1]}, height={image_np.shape[0]}")

        return jsonify({
            "image": img_str,
            "full_image": full_img_str,
            "full_width": image_np.shape[1],
            "full_height": image_np.shape[0],
            "box": [startX, startY, endX, endY],
            "roi_width": endX - startX,
            "roi_height": endY - startY
        })
    except Exception as e:
        print(f"Error in /extract: {str(e)}")
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500


@app.route('/refine', methods=['POST'])
def refine():
    try:
        data = request.json
        full_image_data = data.get('image')
        extracted_image_data = data.get('extractedImage')
        if not full_image_data or not extracted_image_data:
            return jsonify({"error": "Missing image data"}), 400
        points = np.array(data['points'])  # Full-image coordinates
        labels = np.array(data['labels'])  # 1 for include (green), 0 for exclude (red)
        box = data['box']

        # Load images
        full_image = Image.open(io.BytesIO(base64.b64decode(full_image_data))).convert("RGB")
        full_image_np = np.array(full_image)
        extracted_image = Image.open(io.BytesIO(base64.b64decode(extracted_image_data))).convert("RGBA")
        extracted_image_np = np.array(extracted_image)
        initial_mask = extracted_image_np[:, :, 3]  # Alpha channel as initial mask

        full_height, full_width = full_image_np.shape[:2]
        mask = initial_mask.copy()

        # Process green points (include) to add areas
        green_points = points[labels == 1]
        added_area_mask = np.zeros_like(mask, dtype=np.uint8)
        if len(green_points) > 0:
            if len(green_points) >= 3:
                hull = cv2.convexHull(green_points.astype(np.int32))
                cv2.fillConvexPoly(added_area_mask, hull, 255)
                print("Added polygon area with convex hull from green points")
            elif len(green_points) == 2:
                point1, point2 = green_points
                x1, y1 = [int(p) for p in point1]
                x2, y2 = [int(p) for p in point2]
                top_left_x = max(0, min(x1, x2) - 20)
                top_left_y = max(0, min(y1, y2) - 20)
                bottom_right_x = min(full_width - 1, max(x1, x2) + 20)
                bottom_right_y = min(full_height - 1, max(y1, y2) + 20)
                cv2.rectangle(added_area_mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
                print(f"Added rectangle from green points: ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")
            else:
                radius = 30
                for point in green_points:
                    x, y = [int(p) for p in point]
                    x = max(0, min(x, full_width - 1))
                    y = max(0, min(y, full_height - 1))
                    cv2.circle(added_area_mask, (x, y), radius, 255, -1)
                    print(f"Added circle at (x={x}, y={y}) with radius {radius}")
        mask = cv2.bitwise_or(mask, added_area_mask)

        # Process red points (exclude) to remove areas as a polygon
        red_points = points[labels == 0]
        removed_area_mask = np.zeros_like(mask, dtype=np.uint8)
        if len(red_points) > 0:
            if len(red_points) >= 3:
                hull = cv2.convexHull(red_points.astype(np.int32))
                cv2.fillConvexPoly(removed_area_mask, hull, 255)
                print("Removed polygon area with convex hull from red points")
            elif len(red_points) == 2:
                point1, point2 = red_points
                x1, y1 = [int(p) for p in point1]
                x2, y2 = [int(p) for p in point2]
                top_left_x = max(0, min(x1, x2) - 20)
                top_left_y = max(0, min(y1, y2) - 20)
                bottom_right_x = min(full_width - 1, max(x1, x2) + 20)
                bottom_right_y = min(full_height - 1, max(y1, y2) + 20)
                cv2.rectangle(removed_area_mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
                print(f"Removed rectangle from red points: ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")
            else:
                radius = 30
                for point in red_points:
                    x, y = [int(p) for p in point]
                    x = max(0, min(x, full_width - 1))
                    y = max(0, min(y, full_height - 1))
                    cv2.circle(removed_area_mask, (x, y), radius, 255, -1)
                    print(f"Removed circle at (x={x}, y={y}) with radius {radius}")
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(removed_area_mask))

        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 1)

        # Apply the updated mask
        full_extracted = cv2.cvtColor(full_image_np, cv2.COLOR_RGB2BGRA)
        full_extracted[:, :, 3] = mask

        extracted_rgb = cv2.cvtColor(full_extracted, cv2.COLOR_BGRA2RGBA)
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