from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn
import base64
from io import BytesIO

app = FastAPI()

# === Paths ===
YOLO_MODEL_PATH = 'best.pt'  # Your YOLOv8 segmentation model
CLASSIFIER_MODEL_PATH = 'resnet18_best.pth'  # Your ResNet18 classifier model

# === Device setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load YOLOv8 model ===
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Load your custom ResNet18 model ===
def get_resnet18():
    model = models.resnet18(pretrained=True)  # Load pretrained weights

    # Freeze feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model

classifier = get_resnet18()
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
classifier.eval().to(device)

# === Albumentations transform ===
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2(),
])

# === Cropping function (your training logic) ===
def crop_with_mask_exact(image_bgr: np.ndarray, masks) -> np.ndarray:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask = masks[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    crop = masked_img[y:y+h, x:x+w]
    return crop

# # === AFB1 severity estimation ===
# def estimate_afb1_severity(image_tensor, threshold_low=0.2, threshold_high=0.5):
#     image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
#     image_np = (image_np * 255).astype(np.uint8)
#     hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

#     # Aspergillus-like yellow/olive-green mold
#     lower_mold_color = np.array([30, 40, 40])
#     upper_mold_color = np.array([90, 255, 255])
#     mask_color = cv2.inRange(hsv, lower_mold_color, upper_mold_color)

#     # Black mold detection
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 80, 80])
#     mask_black = cv2.inRange(hsv, lower_black, upper_black)

#     combined_mask = cv2.bitwise_or(mask_color, mask_black)

#     mold_pixels = np.sum(combined_mask > 0)
#     total_pixels = combined_mask.size
#     mold_ratio = mold_pixels / total_pixels

#     if mold_ratio >= threshold_high:
#         return "High"
#     elif mold_ratio >= threshold_low:
#         return "Moderate"
#     else:
#         return "Low"

def estimate_afb1_severity(image_tensor, threshold_low=0.2, threshold_high=0.5):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    lower_mold_color = np.array([30, 40, 40])
    upper_mold_color = np.array([90, 255, 255])
    mask_color = cv2.inRange(hsv, lower_mold_color, upper_mold_color)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 80, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    combined_mask = cv2.bitwise_or(mask_color, mask_black)

    mold_pixels = np.sum(combined_mask > 0)
    total_pixels = combined_mask.size
    mold_ratio = mold_pixels / total_pixels

    if mold_ratio >= threshold_high:
        severity = "High"
    elif mold_ratio >= threshold_low:
        severity = "Moderate"
    else:
        severity = "Low"

    # Draw contours on the image
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = image_np.copy()
    cv2.drawContours(marked_img, contours, -1, (255, 0, 0), 2)
    cv2.putText(marked_img, f"Severity: {severity}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert to base64
    pil_img = Image.fromarray(marked_img)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return severity, img_base64

# === API Endpoint ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_bgr = cv2.imread(temp_path)
    if image_bgr is None:
        os.remove(temp_path)
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

    results = yolo_model.predict(source=image_bgr, save=False, save_crop=False)

    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            os.remove(temp_path)
            return JSONResponse(content={"error": "No corn ear detected."}, status_code=400)

        # Crop corn ear
        crop_rgb = crop_with_mask_exact(image_bgr, r.masks.data)

        # Preprocess and classify
        transformed_tensor = val_transform(image=crop_rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(transformed_tensor)
            prob = output.item()
            prediction = "afb1" if prob > 0.5 else "healthy"

        os.remove(temp_path)

        if prediction == "afb1":
            severity, marked_image_base64 = estimate_afb1_severity(transformed_tensor.squeeze(0))
            return {
                "prediction": prediction,
                "confidence": round(prob, 4),
                "affected": True,
                "severity": severity,
                "marked_image_base64": marked_image_base64  # PNG image encoded as base64
            }
        else:
            return {
                "prediction": prediction,
                "confidence": round(prob, 4),
                "affected": False
            }

    os.remove(temp_path)
    return JSONResponse(content={"error": "Unexpected error."}, status_code=500)
