import logging
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from train import TrafficSignNet  # Model class

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI()

# Load the model
model = TrafficSignNet()
model.load_state_dict(torch.load("traffic_sign_model.pth"))
model.eval()

# Confidence threshold setting
CONFIDENCE_THRESHOLD = 0.70  # We want the model's prediction to be made with at least 70% confidence

def predict_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))  # Load the image from byte data
        logger.info("Image successfully loaded")

        # Transformations to be applied on the image before prediction
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
        logger.info("Image successfully transformed")

        with torch.no_grad():  # No gradient calculation needed for inference
            output = model(image)  # Get model output
            _, predicted = torch.max(output, 1)  # Get predicted class
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()  # Calculate confidence

        # Check if the confidence is above the threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Low confidence: {confidence}. Result is invalid.")
            return "Unknown"  # If confidence is low, return "Unknown"

        # If confidence is sufficient, return the predicted class (speed limit)
        logger.info(f"Confidence: {confidence}, Model prediction: {predicted.item()}")
        return 20 if predicted.item() == 0 else 30

    except Exception as e:
        logger.error(f"Error in predict function: {e}")
        raise e


@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        logger.info(f"Received file: {file.filename}")
        image_bytes = await file.read()  # Read the uploaded file as bytes
        speed_limit = predict_image(image_bytes)  # Predict the speed limit based on the image
        logger.info(f"Predicted speed limit: {speed_limit}")

        return JSONResponse(content={"speed_limit": speed_limit})  # Return the prediction as a JSON response
    except Exception as e:
        logger.error(f"API error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)  # Return error if prediction fails
