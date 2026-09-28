from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

MODEL_PATH = "skin_disease_model.keras"

# Load AI model
model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully!")


# Load class names
with open("class_names.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

print("Classes:", class_names)


# Disease information
disease_info = {

    "acne": {
        "symptoms": [
            "Pimples",
            "Blackheads or whiteheads",
            "Red skin"
        ],
        "precautions": [
            "Keep skin clean",
            "Do not squeeze pimples",
            "Use gentle skincare"
        ],
        "doctor": "Consult a dermatologist if acne is severe, painful, persistent, or leaving scars."
    },

    "eczema": {
        "symptoms": [
            "Dry skin",
            "Itching",
            "Redness or irritation"
        ],
        "precautions": [
            "Keep skin moisturized",
            "Avoid irritants",
            "Use gentle skincare"
        ],
        "doctor": "Consult a dermatologist if symptoms are severe, persistent, or getting worse."
    },

    "normal": {
        "symptoms": [
            "No obvious abnormal skin changes detected"
        ],
        "precautions": [
            "Maintain skin hygiene",
            "Use sunscreen",
            "Keep skin moisturized"
        ],
        "doctor": "If you notice unusual, persistent, or changing skin changes, consult a dermatologist."
    },

    "psoriasis": {
        "symptoms": [
            "Red patches",
            "Dry or scaly skin",
            "Itching"
        ],
        "precautions": [
            "Keep skin moisturized",
            "Avoid scratching",
            "Avoid known triggers"
        ],
        "doctor": "Consult a dermatologist if patches are spreading, painful, or persistent."
    },

    "rosacea": {
        "symptoms": [
            "Facial redness",
            "Small bumps",
            "Sensitive skin"
        ],
        "precautions": [
            "Use gentle skincare",
            "Protect skin from sunlight",
            "Avoid known triggers"
        ],
        "doctor": "Consult a dermatologist if facial redness or irritation is persistent or worsening."
    },

    "seborrheic_keratosis": {
        "symptoms": [
            "Raised skin growth",
            "Brown or dark patch",
            "Rough or waxy surface"
        ],
        "precautions": [
            "Do not scratch",
            "Monitor changes"
        ],
        "doctor": "Consult a dermatologist if the skin growth changes in size, shape, colour, or appearance."
    },

    "vitiligo": {
        "symptoms": [
            "Light or white patches",
            "Loss of skin colour",
            "Patches may become noticeable"
        ],
        "precautions": [
            "Protect skin from sunlight",
            "Use sunscreen"
        ],
        "doctor": "Consult a dermatologist for proper evaluation, especially if white patches are spreading."
    }
}


# Home route
@app.route("/")
def home():
    return "Skin Disease AI Backend is Running"


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400


        uploaded_file = request.files["image"]


        if uploaded_file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400


        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        print("Original image size:", image.size)


        # Resize image
        image = image.resize((224, 224))


        # Convert image to array
        image_array = np.array(
            image,
            dtype=np.float32
        )


        # MobileNetV2 preprocessing
        image_array = preprocess_input(
            image_array
        )


        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # AI prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )


        prediction_values = predictions[0]


        # Find highest probability
        predicted_index = int(
            np.argmax(prediction_values)
        )


        confidence = float(
            prediction_values[predicted_index]
        ) * 100


        # Get disease name
        if predicted_index < len(class_names):

            disease = class_names[
                predicted_index
            ]

        else:

            disease = "Unknown"


        # Get disease information
        info = disease_info.get(

            disease,

            {
                "symptoms": [
                    "Information unavailable"
                ],

                "precautions": [
                    "Please consult a dermatologist"
                ],

                "doctor":
                "Please consult a qualified dermatologist."
            }
        )


        # Print results in terminal
        print("\n-----------------------------")
        print("Final Prediction:", disease)
        print(
            "Prediction probabilities:",
            prediction_values
        )
        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )
        print("-----------------------------\n")


        # Send result to frontend
        return jsonify({

            "success": True,

            "disease": disease,

            "confidence":
            round(confidence, 2),

            "symptoms":
            info["symptoms"],

            "precautions":
            info["precautions"],

            "doctor":
            info["doctor"]

        })


    except Exception as error:

        print(
            "ERROR:",
            str(error)
        )


        return jsonify({

            "success": False,

            "error":
            str(error)

        }), 500


# Start Flask server
if __name__ == "__main__":

    print(
        "Starting Skin Disease AI Backend..."
    )

