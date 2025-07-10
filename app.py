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
import requests
from bs4 import BeautifulSoup
import random
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# --- Model and Config URLs ---
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# --- File Paths ---
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
YOLO_MODEL_PATH = "yolov8n.pt"

# --- Download Function ---
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {os.path.basename(path)}: {e}")
            exit() # Exit if a critical file can't be downloaded

# --- Download all necessary files ---
download_file(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH)

# Load YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)

# SAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device)
predictor = SamPredictor(sam)

def scrape_ai_ml_facts():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"  # Replace with a real site containing AI/ML facts
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Adjust the selector based on the actual site's structure
        facts = soup.select('p.fact')  # Hypothetical selector for fact paragraphs
        return [fact.get_text().strip() for fact in facts if fact.get_text().strip()]
    except Exception as e:
        print(f"Scraping failed: {str(e)}")
        # Fallback facts in case scraping fails
        return [
            "AI can process data 1000 times faster than a human brain.",
            "Machine Learning models improve with more data over time.",
            "The term 'Artificial Intelligence' was coined in 1956."
        ]
AI_ML_FACTS = [
    "The term 'Artificial Intelligence' was coined by John McCarthy in 1956.",
    "Machine Learning is a subset of AI that allows systems to learn from data.",
    "The first AI program, Logic Theorist, was developed in 1955 by Herbert Simon and Allen Newell.",
    "Deep Learning, a subset of ML, uses neural networks with many layers to analyze data.",
    "In 1997, IBM's Deep Blue defeated world chess champion Garry Kasparov.",
    "AI can process and analyze data 1000 times faster than a human brain.",
    "The global AI market is expected to reach $500 billion by 2024.",
    "Neural networks are inspired by the structure of the human brain.",
    "AI is used in healthcare to predict diseases with up to 90% accuracy.",
    "The first chatbot, ELIZA, was created in 1966 by Joseph Weizenbaum."
]
@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)

    results = model(image_np)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            detections.append({
                "label": label,
                "score": conf,
                "box": [x1, y1, x2, y2]
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
        padding = 0.9
        width = endX - startX
        height = endY - startY
        startX = int(startX + width * (1 - padding) / 2)
        startY = int(startY + height * (1 - padding) / 2)
        endX = int(endX - width * (1 - padding) / 2)
        endY = int(endY - height * (1 - padding) / 2)
        roi = image_np[startY:endY, startX:endX]

        predictor.set_image(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
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

        # Use static facts
        facts = AI_ML_FACTS
        print(f"Sending facts: {facts}")  # Log all facts being sent

        response_data = {
            "image": img_str,
            "full_image": full_img_str,
            "full_width": image_np.shape[1],
            "full_height": image_np.shape[0],
            "box": [startX, startY, endX, endY],
            "roi_width": endX - startX,
            "roi_height": endY - startY,
            "facts": facts
        }
        print(f"Full response data: {response_data.keys()}")  # Log the keys of the response
        return jsonify(response_data)
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
        mask = np.zeros_like(initial_mask, dtype=np.uint8)  # Start with a blank mask

        # Process red points (label 0) to define the base area to keep
        red_points = points[labels == 0]
        if len(red_points) > 0:
            red_area_mask = np.zeros_like(mask, dtype=np.uint8)
            if len(red_points) >= 3:
                hull = cv2.convexHull(red_points.astype(np.int32))
                cv2.fillConvexPoly(red_area_mask, hull, 255)
                print("Defined base mask with convex hull from red points")
            elif len(red_points) == 2:
                point1, point2 = red_points
                x1, y1 = [int(p) for p in point1]
                x2, y2 = [int(p) for p in point2]
                top_left_x = max(0, min(x1, x2) - 20)
                top_left_y = max(0, min(y1, y2) - 20)
                bottom_right_x = min(full_width - 1, max(x1, x2) + 20)
                bottom_right_y = min(full_height - 1, max(y1, y2) + 20)
                cv2.rectangle(red_area_mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
                print(f"Defined base mask with rectangle from red points: ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")
            else:
                radius = 30
                for point in red_points:
                    x, y = [int(p) for p in point]
                    x = max(0, min(x, full_width - 1))
                    y = max(0, min(y, full_height - 1))
                    cv2.circle(red_area_mask, (x, y), radius, 255, -1)
                    print(f"Defined base mask with circle at (x={x}, y={y}) with radius {radius}")
            mask = red_area_mask  # Set the mask to the red-dotted area
        else:
            mask = initial_mask.copy()  # If no red points, start with the original mask
            print("No red points provided, starting with original mask")

        # Process green points (label 1) to add areas to the mask
        green_points = points[labels == 1]
        if len(green_points) > 0:
            added_area_mask = np.zeros_like(mask, dtype=np.uint8)
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
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=True, host='0.0.0.0', port=port)