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
        roi = image_np[startY:endY, startX:endX]  # Extract ROI

        predictor.set_image(roi)
        masks, _, _ = predictor.predict(
            box=np.array([0, 0, roi.shape[1], roi.shape[0]]),
            multimask_output=False,
        )
        mask = masks[0]

        # Convert mask to uint8 and ensure it matches ROI dimensions
        mask = mask.astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (5, 5), 1)
        if mask.shape != roi.shape[:2]:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Apply mask to ROI
        extracted = cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
        extracted[:, :, 3] = mask  # Alpha channel

        # Create a blank canvas the size of the full image
        full_height, full_width = image_np.shape[:2]
        full_extracted = np.zeros((full_height, full_width, 4), dtype=np.uint8)  # BGRA format

        # Place the extracted ROI at the correct position in the full-sized canvas
        full_extracted[startY:endY, startX:endX] = extracted

        # Convert to RGBA for consistency
        extracted_rgb = cv2.cvtColor(full_extracted, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(extracted_rgb)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        if not img_str.strip():
            return jsonify({"error": "Extracted image is blank"}), 400

        # Full image for display
        full_img = Image.fromarray(image_np)
        full_buffered = io.BytesIO()
        full_img.save(full_buffered, format="PNG")
        full_img_str = base64.b64encode(full_buffered.getvalue()).decode("utf-8")

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
        points = np.array(data['points'])
        labels = np.array(data['labels'])
        box = data['box']

        # Load full image
        full_image = Image.open(io.BytesIO(base64.b64decode(full_image_data))).convert("RGB")
        full_image_np = np.array(full_image)
        extracted_image = Image.open(io.BytesIO(base64.b64decode(extracted_image_data))).convert("RGBA")
        extracted_image_np = np.array(extracted_image)
        initial_mask = extracted_image_np[:, :, 3]  # Alpha channel as initial mask

        startX, startY, endX, endY = [int(x) for x in box]  # Ensure integers

        # Log dimensions for debugging
        print(f"Full image dimensions: {full_image_np.shape}")
        print(f"Initial mask dimensions: {initial_mask.shape}")
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"All points: {points}")
        print(f"All labels: {labels}")

        # Create a copy of the initial mask to modify
        mask = initial_mask.copy()

        # Separate green dots (include) and red dots (exclude)
        green_points = points[labels == 1]  # Points with label 1 (include)

        # Log green points
        print(f"Green points: {green_points}")
        print(f"Number of green points: {len(green_points)}")

        # Apply a manual offset to the coordinates (for debugging)
        y_offset = 50  # Adjust this value based on the observed shift
        green_points_adjusted = green_points.copy()
        green_points_adjusted[:, 1] += y_offset  # Add offset to y-coordinates
        print(f"Adjusted green points: {green_points_adjusted}")

        # Create a mask for the area to add (green dots)
        added_area_mask = np.zeros_like(initial_mask, dtype=np.uint8)
        if len(green_points) > 0:
            if len(green_points) >= 3:  # Need at least 3 points to form a convex hull
                # Compute the convex hull of the green points
                hull = cv2.convexHull(green_points_adjusted.astype(np.int32))
                # Fill the convex hull to include the entire area
                cv2.fillConvexPoly(added_area_mask, hull, 255)
                print("Using convex hull to add area")
            elif len(green_points) == 2:  # Special case for exactly 2 points
                # Draw a rectangle between the two points
                point1, point2 = green_points_adjusted
                x1, y1 = [int(p) for p in point1]
                x2, y2 = [int(p) for p in point2]
                # Compute the bounding rectangle
                top_left_x = max(0, min(x1, x2) - 20)  # Add padding
                top_left_y = max(0, min(y1, y2) - 20)
                bottom_right_x = min(full_image_np.shape[1] - 1, max(x1, x2) + 20)
                bottom_right_y = min(full_image_np.shape[0] - 1, max(y1, y2) + 20)
                # Draw a filled rectangle
                cv2.rectangle(added_area_mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
                print(f"Drawing rectangle from ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")
            else:  # 1 point
                # Draw a circle around the single point
                radius = 30
                for point in green_points_adjusted:
                    x, y = [int(p) for p in point]
                    x = max(0, min(x, full_image_np.shape[1] - 1))
                    y = max(0, min(y, full_image_np.shape[0] - 1))
                    print(f"Drawing circle at (x={x}, y={y}) with radius {radius}")
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                px = int(x + dx)
                                py = int(y + dy)
                                if 0 <= px < full_image_np.shape[1] and 0 <= py < full_image_np.shape[0]:
                                    added_area_mask[py, px] = 255

        # Log the number of non-zero pixels in added_area_mask to verify it’s not empty
        print(f"Non-zero pixels in added_area_mask: {np.count_nonzero(added_area_mask)}")

        # Merge the added area with the initial mask (logical OR)
        mask = cv2.bitwise_or(mask, added_area_mask)

        # Process red dots (exclude)
        for point, label in zip(points, labels):
            if label == 0:  # Only process red dots (exclude)
                x, y = [int(p) for p in point]
                x = max(0, min(x, full_image_np.shape[1] - 1))
                y = max(0, min(y, full_image_np.shape[0] - 1))
                y += y_offset  # Apply the same offset to red dots
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            px = int(x + dx)
                            py = int(y + dy)
                            if 0 <= px < full_image_np.shape[1] and 0 <= py < full_image_np.shape[0]:
                                mask[py, px] = 0  # Set to fully transparent

        # Apply a slight blur to smooth the edges
        mask = cv2.GaussianBlur(mask, (5, 5), 1)

        # Apply the refined mask to the full image
        extracted = cv2.cvtColor(full_image_np, cv2.COLOR_RGB2BGRA)
        extracted[:, :, 3] = mask  # Apply the updated mask

        # Convert back to RGBA for consistency
        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(extracted_rgb)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        img_str = base64.b64encode(img_data).decode("utf-8")  # Encode to base64
        return jsonify({"image": img_str})  # Return as JSON
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