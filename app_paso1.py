
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = load_model("FundusDetection_Model.h5")
IMG_SIZE = (128, 128)

def preprocesar_imagen(image_path, img_size=(128, 128)):
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            print("❌ Error: no se pudo cargar la imagen.")
            return None
        image = cv2.resize(image, img_size)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print("❌ Error procesando imagen:", e)
        return None

def detectar_fondo_de_ojo(image_path, model, umbral=0.5):
    img = preprocesar_imagen(image_path)
    if img is None:
        return False
    pred = model.predict(img)[0][0]
    print(f"📊 Valor de predicción: {pred:.4f}")
    return pred <= umbral

@app.route("/evaluar-imagen", methods=["POST"])
def evaluar_imagen():
    if "imagen" not in request.files:
        return jsonify({"valida": False})

    imagen = request.files["imagen"]
    temp_path = "temp_img.jpg"
    imagen.save(temp_path)

    es_valida = detectar_fondo_de_ojo(temp_path, model)
    os.remove(temp_path)

    return jsonify({"valida": bool(es_valida)})

if __name__ == "__main__":
    app.run(debug=True)
