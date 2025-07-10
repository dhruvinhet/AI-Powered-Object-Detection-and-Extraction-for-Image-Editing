
# 🧠 AI-Powered Object Detection & Extraction for Image Editing

This project combines **YOLOv3 object detection** with the **Segment Anything Model (SAM)** to enable intelligent object extraction and editing from images through a web interface.

---

## 🔧 Setup Instructions

### 📦 Download Required Files

- **YOLOv3 Weights**  
  [➡️ yolov3.weights](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights)

- **SAM Checkpoint**  
  [➡️ sam_vit_h_4b8939.pth](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth)

---

### ⚙️ Backend Setup (Flask)

From the main project directory, install the dependencies and start the server:

```bash
pip install Flask Flask-Cors
python app.py
```

---

### 💻 Frontend Setup (React)

Navigate to the `image-editor/src` directory and run:

```bash
npm install axios @mui/material @emotion/react @emotion/styled @mui/icons-material
npm start
```

---

## 🧭 Visual Overview

An overview of how the system works from detection to editing:

![Screenshot 2025-07-10 140731](https://github.com/user-attachments/assets/0bd4345b-b6c6-4cb4-bc31-7546042b2e47)

---

## 🌐 Frontend Web Interface

How users interact with the app through the browser:

![Screenshot 2025-07-10 140922](https://github.com/user-attachments/assets/a84df7e1-198d-46f1-8819-b79f680f1347)

---

## 🔁 Flask `/detect` Route Flow

Breakdown of the backend logic for handling object detection:

![Screenshot 2025-07-10 141038](https://github.com/user-attachments/assets/e5fef685-1115-47b8-acb7-8ed07b2ebc7d)

---

## ✂️ Image Processing in `/extract` Function

Key steps involved in processing the image after mask extraction:

![Screenshot 2025-07-10 141241](https://github.com/user-attachments/assets/484dcab5-ef92-4047-a140-be568745e704)

---

## 🔄 Refinement Feedback System

Flow for how users can refine the result through feedback:

![Screenshot 2025-07-10 141241](https://github.com/user-attachments/assets/cf43891a-78e1-498b-8387-80236619fcd8)
