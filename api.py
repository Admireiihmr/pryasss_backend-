from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import os
import urllib.request

app = FastAPI()

class ImageInput(BaseModel):
    file: str

# Download model from Hugging Face if not present
MODEL_PATH = "model_weights.keras"
MODEL_URL = "https://huggingface.co/akhilarayampalli/Prayaas/resolve/main/model_weights.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

lm = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict")
async def predict(image_data: ImageInput):
    img_bytes = base64.b64decode(image_data.file)
    img_stream = io.BytesIO(img_bytes)
    img = Image.open(img_stream).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    pred = lm.predict(img_array)
    pred_percent = pred * 100
    results = pred_percent.tolist()[0]
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001)
