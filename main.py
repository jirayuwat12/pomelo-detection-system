import datetime
import json
import os
import uuid
from contextlib import asynccontextmanager

import cv2
import dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator

from pomelo_utils.bytes_processing import bytes_to_cv2_image

# Load environment variables from .env file
if not os.getenv("PROD") == "1":
    print("Loading environment variables from .env file")
    dotenv.load_dotenv()
else:
    print("Loading environment variables from system")

# Create an constants
created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if "TEST_TIME_IMAGES_PATH" not in os.environ:
    os.environ["TEST_TIME_IMAGES_PATH"] = "./test_time_images/"
os.makedirs(os.environ["TEST_TIME_IMAGES_PATH"], exist_ok=True)


# Define a lifespan context to load and unload the model
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before the app starts, load the model
    if os.getenv("MODEL_TYPE") == "random":
        from pomelo_detection.random.random_detector import RandomDetector
        from pomelo_detection.random.random_detector_config import RandomDetectorConfig

        # Load the model
        detector = RandomDetector(RandomDetectorConfig())
        detector.load_model()
        app.state.detector = detector

    yield

    # Stop the app and unload the model
    app.state.detector = None


# Initialize FastAPI app
app = FastAPI(debug=os.getenv("DEBUG", "false").lower() == "true", lifespan=lifespan)

# Instrument the FastAPI app with Prometheus
Instrumentator().instrument(app).expose(app)

# Define a prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to make predictions on an uploaded image.
    """
    # Generate a unique ID for the request
    generated_uuid = str(uuid.uuid4())
    # Read the image file
    contents = await file.read()

    # Convert the bytes to a cv2 image
    image = bytes_to_cv2_image(contents)

    # Get the detector from the app state
    detector = app.state.detector

    # Make a prediction
    result = detector.predict(image)

    # Save the image to the test time images path
    image_path = os.path.join(os.environ["TEST_TIME_IMAGES_PATH"], f"{generated_uuid}.jpg")
    result.image_path = image_path
    json_path = os.path.join(os.environ["TEST_TIME_IMAGES_PATH"], f"{generated_uuid}.json")
    with open(json_path, "w") as json_file:
        json.dump(result.to_json(), json_file)
    cv2.imwrite(image_path, image)

    return result.to_json()


# Define a info endpoint
@app.get("/info/")
async def info():
    """
    Endpoint to get information about the detector.
    """
    # Get the detector from the app state
    detector = app.state.detector

    # Get the detector information
    return {
        "model_type": detector.__class__.__name__,
        "created_at": created_at,
        "test_time_images_path": os.environ["TEST_TIME_IMAGES_PATH"],
    }

# Run
if __name__ == "__main__":
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Add log handler to log to a file
    file_handler = logging.FileHandler(os.getenv("LOG_FILE", "app.log"))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Starting FastAPI server...")

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
