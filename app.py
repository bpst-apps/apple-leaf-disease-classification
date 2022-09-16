# Importing required packages
import os
import numpy as np
import imageio as iio
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, url_for, render_template

# Load pre-trained model
pretrained_model = tf.keras.models.load_model('models/')

# Create flask application
app = Flask(__name__)


def decode_img(image):
    img = tf.image.resize(image, [224, 224])
    img = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img, axis=0)


def make_predictions(image_name):
    classes = ['angular_leaf_spot', 'bean_rust', 'healthy']
    label = np.argmax(pretrained_model.predict(decode_img(iio.imread(f'uploads/{image_name}'))), axis=1)
    return ' '.join(classes[label[0]].split('_')).title()


@app.route('/')
def index():
    return render_template('leaf-disease.html')


@app.route('/api/image', methods=['POST'])
def upload_image():
    # check if upload directory exits
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # check for image
    if 'image' not in request.files:
        return render_template('leaf-disease.html', prediction='No image uploaded!!!')
    file = request.files['image']
    if file.filename == '':
        return render_template('leaf-disease.html', prediction='No image selected!!!')
    file_name = secure_filename(file.filename)
    file.save(os.path.join('uploads', file_name))

    # make predictions
    predictions = make_predictions(file_name)
    predictions = f'Apple Leaf is ' + (predictions if predictions.lower() == 'healthy' else f'having {predictions} Disease')

    return render_template('leaf-disease.html', prediction=predictions)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
