import os
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "DR_grading_best_model_1024.h5"
model = load_model(MODEL_PATH)

# Define allowed extensions for image uploads
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Uploads folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Updated class labels
CLASS_NAMES = ["Not Affected", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Route for home page
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if file was uploaded
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]

        # Check for valid file
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process image
            img_array = preprocess_image(file_path)

            # Make prediction
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            class_name = CLASS_NAMES[class_idx]
            confidence = round(float(predictions[0][class_idx]) * 100, 2)

            # Display result
            if class_idx == 0:
                result = "Not Affected"
            else:
                result = f"Affected: Yes, Class: {class_name}, Confidence: {confidence}%"

            return render_template("index.html", filename=filename, result=result)

    return render_template("index.html", error=None)

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# API endpoint to get JSON response
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process image
        img_array = preprocess_image(file_path)

        # Make prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        class_name = CLASS_NAMES[class_idx]
        confidence = round(float(predictions[0][class_idx]) * 100, 2)

        # Construct JSON response
        if class_idx == 0:
            response = {"affected": "No", "class": "Not Affected"}
        else:
            response = {
                "affected": "Yes",
                "class": class_name,
                "confidence": f"{confidence}%",
            }

        return jsonify(response), 200

    return jsonify({"error": "Invalid file type"}), 400

# Route for the developers page
@app.route("/developers")
def developers():
    return render_template("developers.html")


if __name__ == "__main__":
    app.run(debug=True)
