from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import os
import time
from io import BytesIO
from PIL import Image
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load YOLO model
labelsPath = 'D:/object detection/backend/yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = 'D:/object detection/backend/yolo-coco/yolov3.weights'
configPath = 'D:/object detection/backend/yolo-coco/yolov3.cfg'
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image was properly uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the image
    file = request.files['image']
    image = np.array(Image.open(file.stream))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    (H, W) = image.shape[:2]

    # YOLO prediction
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print(f"[INFO] YOLO took {end - start:.6f} seconds")

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    predictions = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            predictions.append({
                "label": LABELS[classIDs[i]],
                "confidence": confidences[i],
                "box": [x, y, x + w, y + h]
            })

    # Convert the image back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Prepare the image response
    img_io = BytesIO()
    pil_image.save(img_io, 'JPEG', quality=95)
    img_io.seek(0)

    return jsonify({
        "predictions": predictions,
        "image": "data:image/jpeg;base64," + base64.b64encode(img_io.getvalue()).decode('utf-8')
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
