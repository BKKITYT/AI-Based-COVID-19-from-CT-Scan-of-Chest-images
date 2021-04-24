import base64
import numpy as np
import io
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
app = Flask(__name__)
model = load_model('covid_detection.h5')
def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((250, 250))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 250, 250, 1)
    im2arr = im2arr.astype('float32')
    im2arr /= 255 - 0.5
    return im2arr
@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image).tolist()
    check = prediction[0][0]
    check = check * 100
    if check > 40.0:
        response = {'prediction':'High Risk Of Infection'}
    else:
        response = {'prediction':'Low Risk Of Infection'}
    return jsonify(response)
#app.run(port=9000)