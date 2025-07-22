from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

def estimate_afb1_severity(crop_tensor: torch.Tensor, seg_mask: np.ndarray = None, show_mask=False):
    image_np = (crop_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 20, 10])
    upper_green = np.array([95, 255, 180])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 80, 80])
    lower_brown = np.array([5, 30, 10])
    upper_brown = np.array([30, 255, 180])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    combined_mask = cv2.bitwise_or(mask_green, mask_black)
    combined_mask = cv2.bitwise_or(combined_mask, mask_brown)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    if seg_mask is not None:
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_mask = seg_mask > 0
    else:
        valid_mask = np.ones_like(combined_mask, dtype=bool)

    mold_pixels = np.sum((combined_mask > 0) & valid_mask)
    total_pixels = np.sum(valid_mask)

    if total_pixels == 0:
        return "Error: Empty corn area", None, 0.0

    mold_ratio = mold_pixels / total_pixels
    mold_percentage = round(mold_ratio * 100, 2)

    if mold_ratio >= 0.5:
        severity = "High"
    elif mold_ratio >= 0.2:
        severity = "Moderate"
    else:
        severity = "Low"

    # Visualization (optional base64 output)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = image_np.copy()
    cv2.drawContours(marked_img, contours, -1, (255, 0, 0), 2)
    cv2.putText(marked_img, f"Severity: {severity}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    pil_img = Image.fromarray(marked_img)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return severity, img_base64, mold_percentage

# === Update crop function to return both crop image and crop mask ===
def crop_with_mask_exact(image_bgr: np.ndarray, masks) -> tuple[np.ndarray, np.ndarray]:
    mask = masks[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    crop = masked_img[y:y+h, x:x+w]
    crop_mask = mask[y:y+h, x:x+w]

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_mask = (crop_mask > 0).astype(np.uint8)
    return crop_rgb, crop_mask

# === Update predict endpoint logic ===
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

        # Crop both image and mask
        crop_rgb, crop_mask = crop_with_mask_exact(image_bgr, r.masks.data)

        # Preprocess and classify
        transformed_tensor = val_transform(image=crop_rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(transformed_tensor)
            prob = output.item()
            prediction = "afb1" if prob > 0.5 else "healthy"

        os.remove(temp_path)

        if prediction == "afb1":
            # Pass original size crop tensor and mask to severity estimation
            crop_tensor = torch.tensor(crop_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).contiguous()
            severity, marked_image_base64, mold_percentage = estimate_afb1_severity(
                crop_tensor, seg_mask=crop_mask
            )
            return {
                "prediction": prediction,
                "confidence": round(prob, 4),
                "affected": True,
                "mold_percentage": mold_percentage,
                "severity": severity,
                "marked_image_base64": marked_image_base64
            }
        else:
            return {
                "prediction": prediction,
                "confidence": round(prob, 4),
                "affected": False
            }

    os.remove(temp_path)
    return JSONResponse(content={"error": "Unexpected error."}, status_code=500)

