import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- Configuration ---
# Set the path to the image you want to classify
IMAGE_PATH = 'C:/Users/moham/OneDrive/Desktop/cats and dogs/test/cats/cat_96.jpg'
TARGET_SIZE = (224, 224) # MobileNetV2 is typically trained with 224x224 input
TOP_PREDICTIONS = 5 # Number of top predictions to show

# --- Load Model and Image ---
try:
    # Load the MobileNetV2 model pre-trained on ImageNet
    model = MobileNetV2(weights='imagenet')
    print("MobileNetV2 model loaded successfully.")

    # Load the image using Keras utility, resize it to the target size
    img = image.load_img(IMAGE_PATH, target_size=TARGET_SIZE)
    print(f"Image loaded and resized to {TARGET_SIZE}.")

    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)

    # The model expects a batch of images, so we expand the dimensions
    # to create a batch size of 1: (height, width, channels) -> (1, height, width, channels)
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocess the image for MobileNetV2
    # This scales the pixel values from [0, 255] to [-1, 1] as expected by the model
    processed_image = preprocess_input(img_batch)

    # --- Make Prediction ---
    # Get the model's predictions (a vector of probabilities for the 1000 classes)
    predictions = model.predict(processed_image)

    # Decode the predictions into human-readable labels
    # 'decode_predictions' takes the predictions and returns a list of top N tuples:
    # [(class_id, class_name, probability), ...]
    decoded_predictions = decode_predictions(predictions, top=TOP_PREDICTIONS)[0]

    # --- Output Results ---
    print("\n--- Image Classification Results ---")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"Top {i+1}: {label} ({score:.2f})")

except FileNotFoundError:
    print(f"\nError: Image file not found at '{IMAGE_PATH}'. Please update IMAGE_PATH to your image location.")
except ImportError as e:
    print(f"\nError: Required library not found. Please ensure you have the necessary libraries installed (tensorflow, numpy, opencv-python). Detail: {e}")
except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")

# Note: If you are using your *own trained model* (not pre-trained on ImageNet), 
# you would replace `MobileNetV2(weights='imagenet')` with `load_model('path/to/your/model.h5')`
# and replace `decode_predictions` with your custom class label mapping.