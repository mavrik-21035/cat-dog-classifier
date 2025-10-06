import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

IMAGE_PATH = 'C:/Users/moham/OneDrive/Desktop/cats and dogs/test/cats/cat_96.jpg'
TARGET_SIZE = (224, 224) 
TOP_PREDICTIONS = 5 

try:
    model = MobileNetV2(weights='imagenet')
    print("MobileNetV2 model loaded successfully.")
    img = image.load_img(IMAGE_PATH, target_size=TARGET_SIZE)
    print(f"Image loaded and resized to {TARGET_SIZE}.")
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=0)
    processed_image = preprocess_input(img_batch)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=TOP_PREDICTIONS)[0]
    print("\n--- Image Classification Results ---")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"Top {i+1}: {label} ({score:.2f})")

except FileNotFoundError:
    print(f"\nError: Image file not found at '{IMAGE_PATH}'. Please update IMAGE_PATH to your image location.")
except ImportError as e:
    print(f"\nError: Required library not found. Please ensure you have the necessary libraries installed (tensorflow, numpy, opencv-python). Detail: {e}")
except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")
