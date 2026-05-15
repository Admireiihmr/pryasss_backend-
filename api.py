from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import os
import urllib.request

MODEL_PATH = "model_weights.keras"
MODEL_URL = "https://huggingface.co/akhilarayampalli/Prayaas/resolve/main/model_weights.keras"

lm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lm
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found locally. Downloading from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"Download complete. File size: {os.path.getsize(MODEL_PATH)} bytes")
        except Exception as e:
            print(f"Download FAILED: {e}")
            raise
    else:
        print(f"Model found locally. File size: {os.path.getsize(MODEL_PATH)} bytes")

    print("Loading model...")
    lm = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    yield

app = FastAPI(lifespan=lifespan)

class ImageInput(BaseModel):
    file: str

@app.post("/predict")
async def predict(image_data: ImageInput):
    img_bytes = base64.b64decode(image_data.file)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_array = img_to_array(img.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred = lm.predict(img_array)
    return {"predictions": (pred * 100).tolist()[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001)
