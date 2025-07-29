from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
import base64
import os

app = FastAPI()

class ImageInput(BaseModel):
    file: str  # base64 string

# Load the model architecture
with open('model.json', 'r') as json_file:
    model_json = json_file.read()

lm = model_from_json(model_json)
lm.load_weights('model_weights.keras')
lm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

@app.post("/predict")
async def predict(image_data: ImageInput):
    img_bytes = base64.b64decode(image_data.file)
    img = Image.open(io.BytesIO(img_bytes)).resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    pred = lm.predict(img_array)
    pred_percent = pred * 100
    results = [item for sublist in pred_percent.tolist() for item in sublist]
    return {"prediction": results}

# Important: Use $PORT from Render environment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
