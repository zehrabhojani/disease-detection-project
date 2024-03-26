
# from flask import Flask, render_template, request, redirect, url_for
# import numpy as np
# import cv2
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split

# app = Flask(__name__)

# # Load and preprocess X-ray images
# def preprocess_image(file_storage):
#     image_stream = file_storage.read()
#     nparr = np.fromstring(image_stream, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     resized_image = cv2.resize(image, (100, 100))
#     gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#     return gray_image.flatten()

# #-- Assuming you have a dataset with X-ray images and corresponding labels
# # --Load and preprocess the dataset here
# # --X, y = load_and_preprocess_dataset()

# # --Split dataset into training and testing sets
# # --X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --Train a classifier
# # --classifier = SVC(kernel='linear')
# # --classifier.fit(X_train, y_train)

# #-- For demonstration, let's use a dummy classifier without training
# classifier = SVC(kernel='linear')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             image = preprocess_image(file)
#             # --Example: Predict using the loaded model
#             #-- prediction = classifier.predict([image])
#             # --For demonstration, let's assume a prediction
#             prediction = 0  # Dummy prediction
#             if prediction == 0:
#                 result = "Normal"
#             else:
#                 result = "Disease Detected"
#             return render_template('result.html', result=result)
#     return redirect(url_for('index'))

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, redirect, url_for
# import cv2
# import numpy as np
# from yolov3 import YOLOv3  # Assuming you have a YOLOv3 implementation
# from PIL import Image

# app = Flask(__name__)

# # Load the YOLO model
# yolo_model = YOLOv3(weights='disease.weights')

# def preprocess_image(file_storage):
#     image_stream = file_storage.read()
#     image = Image.open(image_stream)
#     return np.array(image)

# def detect_diseases(image):
#     # Perform inference using YOLO model
#     results = yolo_model.detect(image)
#     # Process results to extract detected diseases and bounding boxes
#     # For demonstration, let's assume dummy detection results
#     diseases = ["Normal"]  # Dummy diseases
#     boxes = []  # Dummy bounding boxes
#     return diseases, boxes

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             image = preprocess_image(file)
#             diseases, boxes = detect_diseases(image)
#             return render_template('result.html', diseases=diseases, boxes=boxes)
#     return redirect(url_for('index'))

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import torch
from PIL import Image
import io
from torchvision.transforms import functional as F
from yolov3 import YOLOv3  # Assuming you have a YOLOv3 implementation

app = Flask(__name__)

# Load the YOLO model
yolo_model = YOLOv3(weights='disease.weights')

def preprocess_image(file_storage):
    image_stream = file_storage.read()
    image = Image.open(io.BytesIO(image_stream))
    # Preprocess the image
    image = F.resize(image, (416, 416))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming normalization
    return image.unsqueeze(0)

# def detect_diseases(image):
#     # Perform inference using YOLO model
#     results = yolo_model.detect(image)
#     # Process results to extract detected diseases and bounding boxes
#     # For demonstration, let's assume dummy detection results
#     diseases = ["Normal"]  # Dummy diseases
#     boxes = []  # Dummy bounding boxes
#     return diseases, boxes

def detect_diseases(image):
    # Perform inference using YOLO model
    results = yolo_model.detect(image)
    
    # Check if results are None
    if results is None:
        # If no objects are detected, return empty lists
        return [], []

    # Extract detected classes and bounding boxes from results
    detected_classes = []
    detected_boxes = []
    
    for detection in results:
        class_id, confidence, box = detection
        detected_classes.append(class_id)
        detected_boxes.append(box)
    
    # Map class IDs to human-readable labels (e.g., "Normal" or "Disease")
    class_labels = ["Normal", "Disease"]  # Example labels
    
    # Convert class IDs to human-readable labels
    diseases = [class_labels[class_id] for class_id in detected_classes]
    
    return diseases, detected_boxes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = preprocess_image(file)
            diseases, boxes = detect_diseases(image)
            return render_template('result.html', diseases=diseases, boxes=boxes)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
