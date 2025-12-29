from fastapi import APIRouter, HTTPException, status
import numpy as np
import os
from typing import List

from ..schemas import PredictionRequest, PredictionResponse
from ..config import settings

router = APIRouter(prefix="/api", tags=["Prediction"])

# Load the ML model at startup
model = None


def load_model():
    """Load the Keras model"""
    global model
    try:
        # Import custom layers - add backed_fast to path first
        import sys
        backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        # Import ALL custom objects
        from CNN import (
            NSLPredictionModel, CustomConv1D, CustomMaxPooling1D, CustomDense,
            custom_relu, custom_softmax, custom_categorical_crossentropy, custom_accuracy
        )
        
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load_model # type: ignore

        if settings.MODEL_PATH.startswith('/'):
            model_path = settings.MODEL_PATH
        else:
            backed_fast_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            model_path = os.path.join(backed_fast_root, settings.MODEL_PATH)

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load model with ALL custom objects
        model = keras_load_model(
            model_path,
            custom_objects={
                "NSLPredictionModel": NSLPredictionModel,
                "CustomConv1D": CustomConv1D,
                "CustomMaxPooling1D": CustomMaxPooling1D,
                "CustomDense": CustomDense,
                "custom_relu": custom_relu,
                "custom_softmax": custom_softmax,
                "custom_categorical_crossentropy": custom_categorical_crossentropy,
                "custom_accuracy": custom_accuracy,
            }
        )
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        model = None


@router.post("/predict", response_model=PredictionResponse)
async def predict_sign(prediction_data: PredictionRequest):
    """Predict sign language character from hand landmarks"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded"
        )

    try:
        # Convert hand landmarks to numpy array
        hand_landmarks_array = np.array(prediction_data.hand_landmarks, dtype=np.float32)

        # Flatten to shape (63,) - 21 landmarks × 3 coordinates
        hand_landmarks_flattened = hand_landmarks_array.flatten()

        # Reshape to (63, 1) for Conv1D input, then add batch dimension -> (1, 63, 1)
        hand_landmarks_reshaped = hand_landmarks_flattened.reshape(63, 1)
        hand_landmarks_reshaped = np.expand_dims(hand_landmarks_reshaped, axis=0)

        # Make prediction
        predictions = model.predict(hand_landmarks_reshaped, verbose=0)

        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # Map class index to character name
        class_names = [
            "Ka", "Kha", "Ga", "Gha", "Nga",
            "Cha", "Chha", "Ja", "Jha", "Yan",
            "Ta", "Tha", "Da", "Dha", "Na",
            "Taa", "Thaa", "Daa", "Dhaa", "Naa",
            "Pa", "Pha", "Ba", "Bha", "Ma",
            "Ya", "Ra", "La", "Wa",
            "T_Sha", "M_Sha", "D_Sha", "Ha",
            "Ksha", "Tra", "Gya"
        ]

        predicted_character = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

        return {
            "prediction": predicted_character,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )
