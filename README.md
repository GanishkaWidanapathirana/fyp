# 🌽 AFB1 Corn Mold Detection API (FastAPI)

This FastAPI application detects **AFB1 mold contamination** in corn images using:

- 🧠 YOLOv8 segmentation to locate and crop corn ears
- 🧬 ResNet-18 transfer learning model to classify AFB1 presence
- 🎨 HSV color analysis to estimate mold severity (Low / Moderate / High)

---

## 🚀 Features

- 🖼 Upload a corn image via REST API
- 🧠 Detect and crop the corn ear using trained YOLOv8 segmentation
- 🔍 Classify the cropped corn as `healthy` or `afb1`-affected
- 🎯 If affected, estimate the **severity** using HSV-based mold segmentation
- 📦 Returns a clean JSON response

---

## 🛠️ Technologies Used

- **FastAPI** – Web API Framework
- **YOLOv8** – Corn ear segmentation
- **ResNet-18 (Transfer Learning)** – AFB1 mold classification
- **Albumentations** – Image preprocessing
- **OpenCV + NumPy** – Image operations & HSV mask logic
- **TorchVision / PyTorch** – Deep Learning

---

## 🧬 Model Files

Place the following trained model files in the root of the project:

| File Name            | Description                          |
|----------------------|--------------------------------------|
| `best.pt`            | YOLOv8 segmentation model (ear crop) |
| `resnet18_best.pth`  | Binary classifier for AFB1 detection |

---

## �� Folder Structure

```
afb1_detection_api/
├── main.py              # FastAPI app
├── best.pt              # YOLOv8 segmentation model
├── resnet18_best.pth    # ResNet-18 classifier model
├── requirements.txt     # All Python dependencies
└── README.md           # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo or create a new folder

```bash
mkdir afb1_detection_api && cd afb1_detection_api
```

### 2. Create virtual environment (recommended)

```bash
python -m venv afb1
source afb1/bin/activate  # Windows: afb1\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API server

```bash
uvicorn main:app --reload
```

---

## 🧪 Testing the API

1. Open your browser at: `http://127.0.0.1:8000/docs`
2. Click the `/predict/` endpoint
3. Upload a `.jpg` or `.png` corn image
4. Click "Execute"

You will receive a response like:

---

## 📤 Sample API Responses

### ✅ Healthy Corn

```json
{
  "prediction": "healthy",
  "confidence": 0.1312,
  "affected": false
}
```

### ⚠️ AFB1-Affected Corn

```json
{
  "prediction": "afb1",
  "confidence": 0.9221,
  "affected": true,
  "mold_pecentage": 37.42,
  "severity": "High",
  "marked_image_base64": "iVBORw0KGgoAAAANSUhEUgAABVYAAAK0CAYAAAC9..."  // shortened for clarity
}
```

## AFB1 Severity Estimation Logic
This system includes a function called estimate_afb1_severity that analyzes an input image to determine the severity of mold contamination based on color segmentation in the HSV color space. The result helps identify the likelihood of Aflatoxin B1 (AFB1) contamination in corn samples.

### 🎯 How It Works
Input: The function accepts a preprocessed image tensor (from a corn crop), typically generated from a YOLOv8 segmentation output.

#### Color Filtering in HSV:

    The image is converted from RGB to HSV color space.

    Two masks are created to detect potential mold areas:

    Mask for mold-like colors: Yellow-greenish tones (H: 30–90, S: 40–255, V: 40–255).

    Mask for black mold: Dark/black areas (H: 0–180, S: 0–80, V: 0–80).

#### Combined Mold Mask:

    Both masks are combined to form one binary mask highlighting mold-affected pixels.

    Mold Coverage Calculation:

    The ratio of mold-affected pixels to total pixels is calculated.

    This ratio is converted to a percentage (mold_percentage).

### 📈 Severity Classification
Based on the mold percentage, the severity of the contamination is classified into three levels:

    Mold Ratio (%)	Severity Level
    < 20%	Low
    20% ≤ ratio < 50%	Moderate
    ≥ 50%	High