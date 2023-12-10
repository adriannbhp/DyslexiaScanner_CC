## Start your code here! ##

from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
# from google.cloud import storage

app = Flask(__name__)

model_path = os.path.join('models', 'dyslexia_scanner.h5')
loaded_model = tf.keras.models.load_model(model_path)

# # Google Cloud Storage configuration
# bucket_name = 'testing_data_cc'  # Replace with your GCS bucket name
# storage_client = storage.Client()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    resized_img = tf.image.resize(img, (256, 256))
    return np.expand_dims(resized_img / 255, 0)

# def upload_to_gcs(file_path, blob_name):
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_filename(file_path)

@app.route('/test_dyslexia', methods=['POST'])
def test_dyslexia():
    try:
        # Get the image file from the POST request
        file = request.files['image']
        file.save('uploaded_image.jpg')

        # Preprocess the image
        processed_image = preprocess_image('uploaded_image.jpg')

        # Make predictions
        prediction = loaded_model.predict(processed_image)
        prediction_value = prediction[0][0]

        # Interpret the prediction
        if prediction_value > 0.5:
            diagnosis = "Dyslexia"
        else:
            diagnosis = "Normal"

        return jsonify({"diagnosis": diagnosis, "confidence": float(prediction_value)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

