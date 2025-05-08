import os
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Initialise Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mri_images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class MRIImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

# Configure upload folder and allowed file extensions
app.config['STATIC_FOLDER'] = os.path.join(os.getcwd(), 'app', 'static')
UPLOAD_FOLDER = os.path.join(app.config['STATIC_FOLDER'], 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load AI model and class labels
model_path = os.path.join(os.path.dirname(__file__), "model", "model1.h5")
model = tf.keras.models.load_model(model_path)
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Helper function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing function for images
def preprocess_image(image_path):
    image_size = (128, 128)
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image {image_path}. Skipping")

        # resize the images and normalise the pixel values to train the model
        # faster and more efficiently
        img_resized = cv2.resize(img, image_size)
        img_normalised = img_resized / 255.0
        img_batch = np.expand_dims(img_normalised, axis=0)

    except Exception as e:
        print(f"Erorr processing image {image_path}: {e}")

    return img_batch

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess and predict
            preprocessed_image = preprocess_image(file_path)
            predictions = model.predict(preprocessed_image)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]

            # Save to the database
            new_image = MRIImage(filename=filename, prediction=predicted_class, timestamp=datetime.now())
            db.session.add(new_image)
            db.session.commit()

            # Render result
            return render_template('results.html', filename=filename, prediction=predicted_class)
    return render_template('index.html')

@app.route('/history')
def history():
    images = MRIImage.query.order_by(MRIImage.timestamp.desc()).all()
    return render_template('history.html', images=images)

@app.route('/dashboard')
def dashboard():
    total_images = MRIImage.query.count()
    tumor_counts = db.session.query(
        MRIImage.prediction, db.func.count(MRIImage.prediction)
    ).group_by(MRIImage.prediction).all()

    data = {
        "total_images": total_images,
        "tumor_counts": tumor_counts
    }
    return render_template('dashboard.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)

