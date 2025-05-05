import os
import sys
import json
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from fruit_classifier import FruitClassifier
try:
    from cnn_classifier import CNNClassifier
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("CNN classifier not available. Install PyTorch to use it.")

app = Flask(__name__)
CORS(app)

# Global variables
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_classifier_model')
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_fruit_classifier_model')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
CNN_CLASSES_PATH = os.path.join(MODEL_DIR, 'cnn_class_indices.json')

# Check if models exist
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Traditional model not found at {MODEL_PATH}. Please train a model first.")

if CNN_AVAILABLE and not os.path.exists(CNN_MODEL_PATH):
    print(f"WARNING: CNN model not found at {CNN_MODEL_PATH}. Please train a CNN model first.")

# Load class indices if available
class_indices = {}
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_labels = {int(k): v for k, v in class_indices.items()}
else:
    print(f"WARNING: Class indices not found at {CLASSES_PATH}.")
    # Default classes for testing
    class_labels = {
        0: "apple", 
        1: "banana", 
        2: "orange",
        3: "strawberry",
        4: "grape"
    }

# Initialize classifiers
traditional_classifier = None
cnn_classifier = None

def load_traditional_model():
    global traditional_classifier
    try:
        # Initialize classifier
        traditional_classifier = FruitClassifier()
        traditional_classifier.load_model(MODEL_PATH)
        return True
    except Exception as e:
        print(f"Error loading traditional model: {e}")
        return False

def load_cnn_model():
    global cnn_classifier
    if not CNN_AVAILABLE:
        print("CNN not available, please install PyTorch.")
        return False
    
    try:
        # Initialize CNN classifier
        cnn_classifier = CNNClassifier()
        cnn_classifier.load_model(CNN_MODEL_PATH)
        return True
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "traditional_model_loaded": traditional_classifier is not None,
        "cnn_model_loaded": cnn_classifier is not None,
        "cnn_available": CNN_AVAILABLE
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Return the list of fruit classes the model can classify"""
    # Try to use CNN classes first, then traditional, then default
    if cnn_classifier is not None and hasattr(cnn_classifier, 'class_names'):
        return jsonify({"classes": cnn_classifier.class_names, "model_type": "cnn"})
    elif traditional_classifier is not None and hasattr(traditional_classifier, 'class_names'):
        return jsonify({"classes": traditional_classifier.class_names, "model_type": "traditional"})
    else:
        return jsonify({"classes": list(class_labels.values()), "model_type": "default"})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Classify and grade a fruit from an image"""
    # Get model type from request (cnn or traditional)
    model_type = request.json.get('model_type', 'auto')
    
    # Try to load appropriate model
    if model_type == 'cnn' or (model_type == 'auto' and CNN_AVAILABLE):
        if not cnn_classifier and not load_cnn_model():
            if not traditional_classifier and not load_traditional_model():
                return jsonify({"error": "No models available. Please train a model first."}), 500
            model_type = 'traditional'
        else:
            model_type = 'cnn'
    else:
        if not traditional_classifier and not load_traditional_model():
            if CNN_AVAILABLE and not cnn_classifier and not load_cnn_model():
                return jsonify({"error": "No models available. Please train a model first."}), 500
            model_type = 'cnn'
        else:
            model_type = 'traditional'
    
    # Select the appropriate classifier
    classifier = cnn_classifier if model_type == 'cnn' else traditional_classifier
    
    # Check if the request contains an image
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Get image data
        image_data = request.json['image']
        
        # Remove base64 prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Convert to numpy array for traditional classifier
        img_array = np.array(img) / 255.0
        
        # Predict fruit class
        probabilities = classifier.predict(img_array if model_type == 'traditional' else img)
        predicted_class_idx = np.argmax(probabilities)
        
        # Get class name based on index
        if hasattr(classifier, 'class_names'):
            predicted_class = classifier.class_names[predicted_class_idx]
        else:
            predicted_class = class_labels.get(predicted_class_idx, f"Class {predicted_class_idx}")
        
        confidence = float(probabilities[predicted_class_idx])
        
        # Grade the fruit
        grade, score, features = classifier.grade_fruit(img, predicted_class)
        
        # Convert numpy floats to Python floats for JSON serialization
        features = {k: float(v) for k, v in features.items()}
        
        # Return the results
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "grade": grade,
            "score": float(score),
            "features": features,
            "model_type": model_type
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/grading-criteria', methods=['GET'])
def get_grading_criteria():
    """Return the grading criteria used for each fruit type"""
    # This could be loaded from a database or a file in a real application
    criteria = {
        "default": {
            "color_threshold": {"A": 0.8, "B": 0.6},
            "size_threshold": {"A": 0.8, "B": 0.6},
            "defect_threshold": {"A": 0.1, "B": 0.3}
        },
        "apple": {
            "color_threshold": {"A": 0.85, "B": 0.7},
            "size_threshold": {"A": 0.75, "B": 0.6},
            "defect_threshold": {"A": 0.05, "B": 0.2}
        },
        "banana": {
            "color_threshold": {"A": 0.8, "B": 0.65},
            "size_threshold": {"A": 0.8, "B": 0.65},
            "defect_threshold": {"A": 0.1, "B": 0.25}
        }
        # Additional fruit-specific criteria can be added here
    }
    
    return jsonify({"grading_criteria": criteria})

if __name__ == '__main__':
    # Try to load models at startup
    load_traditional_model()
    if CNN_AVAILABLE:
        load_cnn_model()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 