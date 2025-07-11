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

```json
{
  "prediction": "afb1",
  "confidence": 0.9214,
  "affected": true,
  "severity": "Moderate"
}
```

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
  "severity": "High"
}
```

