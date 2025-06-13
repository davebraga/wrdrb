import os
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir='/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9'"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
import pickle
import io
import uvicorn



app = FastAPI()

# Carregar modelo e encoders
model = load_model("trained_model.keras")
with open("category_encoder.pkl", "rb") as f:
    category_encoder: LabelEncoder = pickle.load(f)
with open("color_encoder.pkl", "rb") as f:
    color_encoder: LabelEncoder = pickle.load(f)

@app.get("/")
def read_root():
    return {"status": "API de classificação rodando com FastAPI"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((160, 160))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        category_pred, color_pred = model.predict(image_array)
        category = category_encoder.inverse_transform([np.argmax(category_pred)])[0]
        color = color_encoder.inverse_transform([np.argmax(color_pred)])[0]

        return JSONResponse(content={
            "categoria": category,
            "cor": color
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Roda localmente sem precisar chamar uvicorn pelo terminal
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
