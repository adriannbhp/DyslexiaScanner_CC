from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import base64
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


def post_to_firestore(diagnosis, confidence, custom_id=None):
    if custom_id:
        data = {
            "id": int(custom_id),  # Convert custom_id to integer
            "diagnosis": diagnosis,
            "confidence": confidence,
            # Add more fields as needed
        }
        doc_ref = db.collection('dyslexia_data').document(str(custom_id))
        doc_ref.set(data)
        return {"id": int(custom_id), "diagnosis": diagnosis, "confidence": confidence}
    else:

        sequential_id = generate_sequential_id()
        data = {
            "id": sequential_id,
            "diagnosis": diagnosis,
            "confidence": confidence,
            # Add more fields as needed
        }
        doc_ref = db.collection('dyslexia_data').document(str(sequential_id))
        doc_ref.set(data)
        return {"id": sequential_id, "diagnosis": diagnosis, "confidence": confidence}

def generate_sequential_id():
    counter = db.collection('counter').document('dyslexia_counter').get().to_dict()
    if counter is None:
        counter = {"value": 1}
        db.collection('counter').document('dyslexia_counter').set(counter)
    else:
        counter["value"] += 1
        db.collection('counter').document('dyslexia_counter').set(counter)
    return counter["value"]

# Example of a simple sequential ID generator
def generate_sequential_id():
    counter = db.collection('counter').document('dyslexia_counter').get().to_dict()
    if counter is None:
        counter = {"value": 1}
        db.collection('counter').document('dyslexia_counter').set(counter)
    else:
        counter["value"] += 1
        db.collection('counter').document('dyslexia_counter').set(counter)
    return str(counter["value"])


def get_dyslexia_data(doc_id=None):
    dyslexia_data = []

    # Retrieve all data or data for a specific document ID
    collection_ref = db.collection('dyslexia_data')

    if doc_id:
        doc_ref = collection_ref.document(doc_id)
        doc = doc_ref.get()

        if doc.exists:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  # Include document ID in the data
            dyslexia_data.append(doc_data)
        else:
            return None  # Document not found
    else:
        docs = collection_ref.stream()

        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  # Include document ID in the data
            dyslexia_data.append(doc_data)

    return dyslexia_data


# Function to delete dyslexia data by ID
def delete_dyslexia_data(data_id):
    try:
        # Get the data to be deleted before deleting it
        document_ref = db.collection('dyslexia_data').document(data_id)
        deleted_data = document_ref.get().to_dict()

        if deleted_data:
            deleted_data['id'] = data_id

            # Delete document from Firestore collection
            document_ref.delete()

            return {"message": "Data deleted successfully", "deleted_data": deleted_data}
        else:
            return {"error": "Document not found"}
    except Exception as e:
        return {"error": str(e)}

def update_dyslexia_data(data_id, new_image_base64):
    try:
        # Check if the document exists before updating
        document_ref = db.collection('dyslexia_data').document(data_id)
        document = document_ref.get()

        if document.exists:
            # Check if a new image is provided
            if new_image_base64 is not None:
                # Decode base64 and save the image
                new_image_data = base64.b64decode(new_image_base64)
                new_image_path = f'updated_image_{data_id}.jpg'
                with open(new_image_path, 'wb') as new_image_file:
                    new_image_file.write(new_image_data)

                # Preprocess the new image
                processed_new_image = preprocess_image(new_image_path)

                if processed_new_image is not None:
                    print("Processed image shape:", processed_new_image.shape)  # Debugging line

                    try:
                        # Make predictions
                        prediction = loaded_model.predict(processed_new_image)
                        
                        if prediction is not None:
                            new_diagnosis = "Dyslexia" if prediction[0][0] > 0.5 else "No Dyslexia"
                            new_confidence = float(prediction[0][0])

                            # Update document in Firestore collection
                            document_ref.update({
                                "diagnosis": new_diagnosis,
                                "confidence": new_confidence
                            })

                            # Retrieve the updated data
                            updated_document = document_ref.get().to_dict()
                            updated_document['id'] = data_id

                            return {"message": "Image processed successfully", "updated_data": updated_document}
                        else:
                            return {"error": "Prediction result is None"}
                    except Exception as prediction_error:
                        print(f"Error during prediction: {prediction_error}")  # Debugging line
                        return {"error": f"Error during prediction: {str(prediction_error)}"}
                else:
                    return {"error": "Failed to preprocess the image"}
            else:
                return {"error": "New image not provided. Please try again with an image."}
        else:
            return {"error": "Document not found"}
    except Exception as e:
        print(f"Exception occurred: {e}")  # Debugging line
        return {"error": str(e)}
    
# Route to handle POST request
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

        # Post data to Firestore and get the ID
        result = post_to_firestore(diagnosis, float(prediction_value))

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
    
# Route to get dyslexia data (all or by ID)
@app.route('/get_dyslexia_data', methods=['GET'])
def get_dyslexia_data_route():
    try:
        doc_id = request.args.get('id')  # Get the 'id' parameter from the request
        data = get_dyslexia_data(doc_id)

        if data is not None:
            return jsonify({"dyslexia_data": data})
        else:
            return jsonify({"error": "Document not found"}), 404 if doc_id else 400
    except Exception as e:
        return jsonify({"error": str(e)})

# Route to get dyslexia data by ID
@app.route('/get_dyslexia_data_by_id', methods=['GET'])
def get_dyslexia_data_by_id_route():
    try:
        doc_id = request.args.get('id')  # Get the 'id' parameter from the request
        if not doc_id:
            return jsonify({"error": "Missing 'id' parameter"}), 400

        data = get_dyslexia_data(doc_id)

        if data is not None:
            return jsonify({"dyslexia_data": data})
        else:
            return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)})


# Route to delete dyslexia data by ID
@app.route('/delete_dyslexia_data/<string:data_id>', methods=['DELETE'])
def delete_dyslexia_data_route(data_id):
    try:
        # Get the data to be deleted before deleting it
        document_ref = db.collection('dyslexia_data').document(data_id)
        deleted_data = document_ref.get().to_dict()
        if deleted_data:
            deleted_data['id'] = data_id

            # Delete document from Firestore collection
            document_ref.delete()

            return jsonify({"message": "Data deleted successfully", "deleted_data": deleted_data})
        else:
            return jsonify({"error": "Document not found"})

    except Exception as e:
        return jsonify({"error": str(e)})
    
    
# Route to update dyslexia data with image only by ID
@app.route('/update_dyslexia_data/<string:data_id>', methods=['PUT'])
def update_dyslexia_data_route(data_id):
    try:
        # Get new image from the request
        new_image_base64 = request.files.get('image').read()

        # Call the updated method
        result = update_dyslexia_data(data_id, new_image_base64)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)

