from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# CORS para Netlify + GitHub Pages
CORS(app, resources={r"/*": {"origins": ["*"]}})

# ---------------------------------------
# 1. CARGAR MODELO TFLITE
# ---------------------------------------
TFLITE_PATH = "FundusDetection_Model.tflite"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (128, 128)

# ---------------------------------------
# 2. PREPROCESAR IMAGEN
# ---------------------------------------
def preprocesar_imagen(image_path, img_size=(128, 128)):
    try:
        image = cv2.imread(image_path)

        if image is None:
            print("❌ Error: no se pudo leer la imagen.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        return image
    except Exception as e:
        print("Error procesando imagen:", e)
        return None

# ---------------------------------------
# 3. DETECTAR SI ES FONDO DE OJO
# ---------------------------------------
def detectar_fondo_de_ojo(image_path, umbral=0.5):
    img = preprocesar_imagen(image_path)
    if img is None:
        return False

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run model
    interpreter.invoke()

    # Output
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    print(f"📊 Valor de predicción TFLITE: {pred:.4f}")

    return pred <= umbral

# ---------------------------------------
# 4. ENDPOINT
# ---------------------------------------
@app.route("/evaluar-imagen", methods=["POST"])
def evaluar_imagen():
    if "imagen" not in request.files:
        return jsonify({"valida": False})

    imagen = request.files["imagen"]
    temp_path = "temp_img.jpg"
    imagen.save(temp_path)

    es_valida = detectar_fondo_de_ojo(temp_path)
    os.remove(temp_path)

    return jsonify({"valida": bool(es_valida)})

# ---------------------------------------
# 5. HOME (para probar que el backend está vivo)
# ---------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "mensaje": "Backend de validación activo (TFLITE)."})

# ---------------------------------------
# 6. EJECUCIÓN LOCAL
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
