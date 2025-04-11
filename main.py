
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

modelo = tf.keras.models.load_model("modelo_mnist.h5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("L")
    img = ImageOps.invert(img.resize((28, 28)))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediccion = modelo.predict(img)
    etiqueta = int(np.argmax(prediccion))
    confianza = float(prediccion[0][etiqueta])

    return {"etiqueta": etiqueta, "confianza": confianza}
