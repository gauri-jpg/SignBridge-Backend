import json
import os
from io import BytesIO

import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from predictor import SignBridgePredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "signbridge_landmark_best.keras")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")


with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


predictor = SignBridgePredictor(
    MODEL_PATH,
    HAND_MODEL_PATH,
    CLASS_NAMES
)


app = FastAPI(
    title="SignBridge API",
    description="Indian Sign Language Alphabet Recognition API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "SignBridge API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    image_array = np.array(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_array
    )

    result = predictor.detector.detect(mp_image)

    if not result.hand_landmarks:
        return {
            "success": False,
            "letter": None,
            "confidence": 0.0,
            "message": "No hand detected."
        }

    landmarks = result.hand_landmarks[0]

    features = predictor.normalize_landmarks(landmarks)

    features = features.reshape(1, 63)

    prediction = predictor.model.predict(
        features,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(prediction))

    predicted_letter = CLASS_NAMES[predicted_index]

    confidence = float(prediction[predicted_index])

    return {
        "success": True,
        "letter": predicted_letter,
        "confidence": round(confidence, 4),
        "message": "Prediction successful."
    }