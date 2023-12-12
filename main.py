from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import firestore
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


app = Flask(__name__)

firebase_admin.initialize_app(options={
    'credential': firebase_admin.credentials.Certificate(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
})

# Connect to firestore
db=firestore.client()

model_path = os.path.join('models', 'dyslexia_scanner.h5')
loaded_model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    resized_img = tf.image.resize(img, (256, 256))
    return np.expand_dims(resized_img / 255, 0)

def post_to_firestore(diagnosis, confidence):
    data = {
        "diagnosis": diagnosis,
        "confidence": confidence,
        # You can add more fields as needed
    }
    db.collection('dyslexia_data').add(data)

def get_dyslexia_data():
    dyslexia_data = []

    # Retrieve data from Firestore collection
    collection_ref = db.collection('dyslexia_data')
    docs = collection_ref.stream()

    for doc in docs:
        dyslexia_data.append(doc.to_dict())

    return dyslexia_data

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
            diagnosis = "Unfortunately, there is a >50% chance of suffering from dyslexia."
        else:
            diagnosis = "Congratulations, you are normal."

        # Post data to Firestore
        post_to_firestore(diagnosis, float(prediction_value))

        return jsonify({"diagnosis": diagnosis, "confidence": float(prediction_value)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_dyslexia_data', methods=['GET'])
def get_dyslexia_data_route():
    try:
        data = get_dyslexia_data()
        return jsonify({"dyslexia_data": data})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

