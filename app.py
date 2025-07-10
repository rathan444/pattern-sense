from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model_path = os.path.join('model', 'model_cnn.h5')
model = load_model(model_path, compile=False)

# List of class labels
class_labels = ['floral', 'striped', 'plain']  # Update this if needed

# Make sure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Home route
@app.route('/')
def home():
    return render_template('home.html')


# Upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')


# Prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded image with unique filename
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess the image (resize to 224x224)
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Pass data to result page
    image_path = url_for('static', filename=f'uploads/{filename}')
    return render_template('result.html', image_path=image_path, prediction=predicted_class)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
