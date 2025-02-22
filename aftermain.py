import cv2
import numpy as np
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageFilter

# Function to update the displayed image
def update_image():
    global img_display, img_tk
    img_display = img.copy()

    # Apply brightness and contrast adjustments
    enhancer = ImageEnhance.Brightness(img_display)
    img_display = enhancer.enhance(brightness_scale.get())
    enhancer = ImageEnhance.Contrast(img_display)
    img_display = enhancer.enhance(contrast_scale.get())

    # Apply additional adjustments
    img_display = img_display.rotate(rotation_scale.get(), expand=True)
    img_display = img_display.resize((int(img.width * resize_scale.get()), int(img.height * resize_scale.get())), Image.LANCZOS)
    if flip_horizontal.get():
        img_display = ImageOps.mirror(img_display)
    if flip_vertical.get():
        img_display = ImageOps.flip(img_display)
    if sharpen_scale.get() > 1.0:
        enhancer = ImageEnhance.Sharpness(img_display)
        img_display = enhancer.enhance(sharpen_scale.get())
    if blur_scale.get() > 0:
        img_display = img_display.filter(ImageFilter.GaussianBlur(blur_scale.get()))
    enhancer = ImageEnhance.Color(img_display)
    img_display = enhancer.enhance(color_scale.get())

    # Convert to ImageTk format
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.config(image=img_tk)

# Function to save the edited image
def save_image():
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        img_display.save(save_path)
        print(f"Image saved to {save_path}")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load Mask R-CNN network (for segmentation)
net = cv2.dnn.readNetFromTensorflow("mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
                                    "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Load input image
image_path = "input.jpg"  # Ensure your image path is correct
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit(1)
(H, W) = image.shape[:2]

# Create blob from image and perform forward pass for detection and segmentation
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])

# Set detection confidence threshold
conf_threshold = 0.5

# Dictionary to store detections for user selection
detections = {}

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
        detections[i] = {
            "label": label,
            "score": score,
            "box": (startX, startY, endX, endY),
            "mask": masks[i, class_id]
        }
        
        # Draw detection on image
        color = (0, 255, 0)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, f"{label}: {score:.2f}", (startX, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image with detections so the user knows what objects were found
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print detected classes with indexes for selection
print("Detected objects:")
for idx, det in detections.items():
    print(f"ID: {idx} - Label: {det['label']} (Confidence: {det['score']:.2f})")

# Let the user choose which object to extract by entering the detection ID
choice = input("Enter the ID of the object you want to extract: ")

try:
    choice = int(choice)
    if choice not in detections:
        print("Invalid choice. Exiting.")
        exit(1)
except ValueError:
    print("Please enter a valid integer ID.")
    exit(1)

# Extract selected detection details
selected = detections[choice]
(startX, startY, endX, endY) = selected["box"]

# Extract ROI for segmentation mask processing
roi = image[startY:endY, startX:endX]
mask = selected["mask"]

# The mask is output at a fixed size; resize to match ROI dimensions with high-quality interpolation
mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)

# Apply a smoothing filter to the mask to reduce jagged edges
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Threshold the mask to obtain a binary mask
_, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
mask = mask.astype("uint8")

# Create a high-quality extracted image with transparent background
# First, create an empty BGRA image
extracted = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)

# Set pixels outside the mask to be transparent
extracted[mask == 0] = (0, 0, 0, 0)

# Convert the extracted image to PIL format for editing
img = Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGRA2RGBA))

# Create a Tkinter window for image editing
root = Tk()
root.title("Image Editor")

# Display the image
img_tk = ImageTk.PhotoImage(img)
img_label = Label(root, image=img_tk)
img_label.pack()

# Add brightness and contrast sliders
brightness_scale = Scale(root, from_=0.5, to=2.0, resolution=0.1, orient=HORIZONTAL, label="Brightness", command=lambda x: update_image())
brightness_scale.set(1.0)
brightness_scale.pack()

contrast_scale = Scale(root, from_=0.5, to=2.0, resolution=0.1, orient=HORIZONTAL, label="Contrast", command=lambda x: update_image())
contrast_scale.set(1.0)
contrast_scale.pack()

# Add rotation slider
rotation_scale = Scale(root, from_=-180, to=180, resolution=1, orient=HORIZONTAL, label="Rotation", command=lambda x: update_image())
rotation_scale.set(0)
rotation_scale.pack()

# Add resize slider
resize_scale = Scale(root, from_=0.1, to=2.0, resolution=0.1, orient=HORIZONTAL, label="Resize", command=lambda x: update_image())
resize_scale.set(1.0)
resize_scale.pack()

# Add flip options
flip_horizontal = Scale(root, from_=0, to=1, resolution=1, orient=HORIZONTAL, label="Flip Horizontal", command=lambda x: update_image())
flip_horizontal.set(0)
flip_horizontal.pack()

flip_vertical = Scale(root, from_=0, to=1, resolution=1, orient=HORIZONTAL, label="Flip Vertical", command=lambda x: update_image())
flip_vertical.set(0)
flip_vertical.pack()

# Add sharpen slider
sharpen_scale = Scale(root, from_=1.0, to=5.0, resolution=0.1, orient=HORIZONTAL, label="Sharpen", command=lambda x: update_image())
sharpen_scale.set(1.0)
sharpen_scale.pack()

# Add blur slider
blur_scale = Scale(root, from_=0, to=10, resolution=1, orient=HORIZONTAL, label="Blur", command=lambda x: update_image())
blur_scale.set(0)
blur_scale.pack()

# Add color adjustment slider
color_scale = Scale(root, from_=0.0, to=2.0, resolution=0.1, orient=HORIZONTAL, label="Color", command=lambda x: update_image())
color_scale.set(1.0)
color_scale.pack()

# Add a save button
save_button = Button(root, text="Save Image", command=save_image)
save_button.pack()

# Start the Tkinter main loop
root.mainloop()