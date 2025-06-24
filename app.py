from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your trained model once
model = tf.keras.models.load_model('fire_classification_model.keras')

IMG_SIZE = 128

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file.")
        
        # Save uploaded file
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Prepare image and predict
        img = prepare_image(file_path)
        preds = model.predict(img)
        class_idx = np.argmax(preds)
        classes = ['Fire', 'Non-Fire']
        prediction = classes[class_idx]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
