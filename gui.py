from fastapi import FastAPI, UploadFile, File
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

# Define FastAPI app
app = FastAPI()

# Load the trained model
model = None

# Define function to preprocess image
def preprocess_image(file):
    img = Image.open(BytesIO(file))
    img = img.resize((256, 256))  # Resize image to match model input shape
    img_array = img_to_array(img) / 255.0  # Convert image to array and normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Define endpoint for image classification
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    global model
    
    if model is None:
        # Load the model if not already loaded
        model = load_model(r"D:\SATHYABAMA\IDP -Project\Project\skincancer.h5")
        
    try:

        if not file.filename.lower().endswith((".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Only JPEG images are supported")
        
        # Preprocess image
        img_array = preprocess_image(await file.read())
        
        # Perform classification
        prediction = model.predict(img_array)
        
        # Get predicted class
        predicted_class = np.argmax(prediction)
        
        # Define class labels
        class_labels = (["benign","malignant"])
        
        # Return predicted class name
        return {"predicted_class": class_labels[predicted_class]}
    except Exception as e:
        return {"error": str(e)}

