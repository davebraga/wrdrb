from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import pickle
import uvicorn

app = FastAPI()

# Carregar modelo e encoders localmente (ajuste os caminhos conforme seu ambiente)
model = load_model("saved_model")
with open("category_encoder.pkl", "rb") as f:
    category_encoder = pickle.load(f)
with open("color_encoder.pkl", "rb") as f:
    color_encoder = pickle.load(f)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((160, 160))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_array = preprocess_image(image_bytes)

    category_pred, color_pred = model.predict(image_array)
    category = category_encoder.inverse_transform([np.argmax(category_pred)])[0]
    color = color_encoder.inverse_transform([np.argmax(color_pred)])[0]

    return JSONResponse(content={"category": category, "color": color})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
