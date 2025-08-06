# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# import numpy as np
# from PIL import Image, UnidentifiedImageError
# import io
# import base64
# import os
# import tensorflow as tf
# from tensorflow.keras.models import model_from_json

# app = FastAPI(title="Oral Health Image Prediction API")

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all for now; restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Base64 image input model
# class ImageInput(BaseModel):
#     file: str  # base64 encoded image string

# # Load model once at startup
# def load_model():
#     try:
#         with open("model.json", "r") as json_file:
#             model_json = json_file.read()
#         model = model_from_json(model_json)
#         model.load_weights("model_weights.keras")
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#             loss="categorical_crossentropy",
#             metrics=["accuracy"],
#         )
#         return model
#     except Exception as e:
#         raise RuntimeError(f"Model loading failed: {e}")

# # Load model into memory
# model = load_model()

# @app.post("/predict")
# async def predict(image_data: ImageInput):
#     try:
#         # Decode base64 image
#         img_bytes = base64.b64decode(image_data.file)
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         img = img.resize((224, 224))

#         # Convert to array and normalize
#         img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

#         # Make prediction
#         prediction = model.predict(img_array)
#         percent_scores = (prediction[0] * 100).tolist()
#         rounded_scores = [round(score, 2) for score in percent_scores]
#         predicted_class = int(np.argmax(prediction[0]))

#         return {
#             "prediction": rounded_scores,
#             "predicted_class": predicted_class
#         }

#     except UnidentifiedImageError:
#         raise HTTPException(status_code=400, detail="Invalid image format.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# # Optional: Run locally
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)


# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import base64

app = FastAPI()

class ImageInput(BaseModel):
    file: str

# ✅ Load full model (saved as model.keras)
lm = tf.keras.models.load_model("model_weights.keras")

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


