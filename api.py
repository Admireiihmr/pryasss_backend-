from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import base64
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Oral Health Image Prediction API")

# ✅ Add CORS middleware AFTER app is created
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    file: str  # base64 encoded image string

# Load and compile model once during startup
def load_model():
    try:
        with open('model.json', 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights('model_weights.keras')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

model = load_model()

@app.post("/predict")
async def predict(image_data: ImageInput):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_data.file)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))

        # Preprocess image
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        pred_percent = (prediction * 100).tolist()[0]
        pred_percent_rounded = [round(p, 2) for p in pred_percent]

        return {
            "prediction": pred_percent_rounded,
            "predicted_class": int(np.argmax(prediction))
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Run locally (not needed on Render)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
