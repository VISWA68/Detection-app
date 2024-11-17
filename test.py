import requests
import json
from PIL import Image
from io import BytesIO
import base64

# Server URL
url = "http://127.0.0.1:5000/predict"  # Update if the server runs on a different host or port

# Path to the image to test
image_path = "D:/object detection/backend/Object dection using image/images/baggage_claim.jpg"

# Send a POST request with the image
with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    print("[INFO] Sending image to server...")
    response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    data = response.json()
    print("[INFO] Received response from server.")
    
    # Print predictions
    predictions = data["predictions"]
    print("\nPredictions:")
    for i, prediction in enumerate(predictions):
        print(f"  {i + 1}. Label: {prediction['label']}, Confidence: {prediction['confidence']:.4f}, Box: {prediction['box']}")

    # Save and display the annotated image
    if "image" in data:
        print("[INFO] Saving annotated image...")
        base64_image = data["image"].split(",")[1]  # Remove the data URI prefix
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        output_path = "output.jpg"
        image.save(output_path)
        print(f"[INFO] Annotated image saved as {output_path}")
        image.show()
else:
    print(f"[ERROR] Server returned status code {response.status_code}")
    print(response.text)
