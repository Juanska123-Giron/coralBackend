from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
from flask_cors import CORS
from flask_cors import cross_origin
from pymongo import MongoClient
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)

new_model = load_model('models/coralModel3.h5')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha proporcionado ninguna imagen'})

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(img, (256, 256))
    yhat = new_model.predict(np.expand_dims(resize / 255, 0))
    print(yhat)
    resultado = 'Coral verdadera' if yhat > 0.5 else 'Falsa coral'

    client = MongoClient('mongodb+srv://root:12118375@cluster0.fkkqruy.mongodb.net/Prueba?retryWrites=true&w=majority')
    db = client['coralDB']
    collection = db['predicciones']


    # Convertir imagen a base64
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Obtener fecha y hora actual
    fecha_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    client = MongoClient('mongodb+srv://root:12118375@cluster0.fkkqruy.mongodb.net/Prueba?retryWrites=true&w=majority')
    db = client['coralDB']
    collection = db['predicciones']

    # Crear un nuevo registro
    registro = {
        'image_path': img_base64,
        'resultado': resultado,
        'fecha': fecha_actual
    }


    # Insertar el registro en la colección
    collection.insert_one(registro)

    return jsonify({'resultado': resultado})


@app.route('/api', methods=['GET'])
@cross_origin()
def api():
    return {
        'backend':'api'
    }

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return {
        'backend':'index'
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5070)
