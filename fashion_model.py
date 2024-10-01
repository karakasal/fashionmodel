import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify
from PIL import Image

# Loading the model
with open('assets\\fashion_model.json', 'r') as file:
    model_json = file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights('assets\\fashion_model.weights.h5')

# Creating Flask
app = Flask(__name__)

@app.route('/img/<string:img_num>', methods=['GET', 'POST'])
def classify(img_num):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    upload_dir = 'assets\\uploads\\'
    image_dir = upload_dir+img_num+'.png'
    img = load_img(image_dir, target_size=(28, 28))
    x = img_to_array(img)
    x = np.mean(x, axis=2)
    x = x.reshape(-1,28*28)
    prediction = model.predict(x)
    im = Image.open(image_dir)
    
    return jsonify({'Image is ': classes[np.argmax(prediction[0])]}), im.show()

app.run(debug=True)