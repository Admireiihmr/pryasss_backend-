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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights.keras")
JSON_PATH = os.path.join(BASE_DIR, "model.json")
MODEL_URL = "https://huggingface.co/akhilarayampalli/Prayaas/resolve/main/model_weights.keras"
JSON_URL = "https://huggingface.co/akhilarayampalli/Prayaas/resolve/main/model.json"

lm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lm

    if not os.path.exists(MODEL_PATH):
        print("Downloading weights...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Weights downloaded. Size: {os.path.getsize(MODEL_PATH)} bytes")

    if not os.path.exists(JSON_PATH):
        print("Downloading model architecture...")
        urllib.request.urlretrieve(JSON_URL, JSON_PATH)
        print("Architecture downloaded.")

    print("Loading model...")
with open(JSON_PATH, "r") as f:
    model_json = f.read()
lm = tf.keras.models.model_from_json(model_json)
lm.load_weights(MODEL_PATH)
lm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print("Model loaded successfully.")

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
