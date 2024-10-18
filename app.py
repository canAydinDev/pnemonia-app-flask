import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image
import cv2

import numpy as np
import threading
import uuid
import time

app = Flask(__name__)
CORS(app)

# Dosya yükleme klasörü
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Modeli yükle
base_model = VGG19(include_top=False, input_shape=(128,128,3))
x = base_model.output
flat=Flatten()(x)
class_1 = Dense(4606, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
my_model = Model(base_model.inputs, output)
my_model.load_weights('vgg_final.weights.h5')

# İşlerin durumu
jobs = {}

# Ana sayfa
@app.route('/')
def home():
    return render_template('home.html')


def long_running_prediction(job_id, img_path):
    try:
        # Resmi işleme
        pic = image.load_img(img_path, target_size=(128, 128))
        pic = np.array(pic)  # Keras'tan yüklenen resmi numpy array'e dönüştür
        pic = np.expand_dims(pic, axis=0)  # Giriş için uygun boyuta genişlet
        pic = preprocess_input(pic)  # Resmi ön işleme tabi tut

        # Tahmin yap
        result = my_model.predict(pic)
        result01 = np.argmax(result, axis=1)

        # İşlem tamamlandı
        jobs[job_id] = {'status': 'completed', 'prediction': int(result01[0])}
    except Exception as e:
        jobs[job_id] = {'status': 'failed', 'error': str(e)}


@app.route('/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return jsonify({'error': 'No file part found'}), 400
    
    img_file = request.files['img']
    
    if img_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Dosya yolunu güvenli bir şekilde oluştur ve yükle
    filename = secure_filename(img_file.filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Klasör yoksa oluştur
    
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_path)

    # İş kimliği oluştur ve arka planda tahmin işlemi başlat
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'pending'}

    # Arka planda işlemi başlat
    threading.Thread(target=long_running_prediction, args=(job_id, img_path)).start()

    # İş kimliğini döndür
    return jsonify({'jobId': job_id})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if job:
        return jsonify(job)
    else:
        return jsonify({'error': 'Job not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
