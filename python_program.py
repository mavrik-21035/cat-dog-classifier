import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 # <-- The key to the high accuracy!

# --- 0. Configuration and Environment Setup ---
# 
# !!! IMPORTANT: YOU MUST CHANGE THIS PATH !!!
# Ensure you use the 'r' prefix for raw string to avoid the Unicode error.
BASE_DIR =r"C:\Users\moham\OneDrive\Desktop\cats and dogs"

# Image and training parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
NUM_CLASSES = 2 # Cat and Dog
EPOCHS = 30 # Increased epochs slightly, can go up to 50 if needed

# --- 1. Data Preparation and Loading ---

# Use heavy Data Augmentation to prevent overfitting on the small dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test data (no augmentation)
test_val_datagen = ImageDataGenerator(rescale=1./255)

# 1.1 Training Data Loader (222 images)
print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    f'{BASE_DIR}/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# 1.2 Validation Data Loader (30 images)
print("Loading Validation Data...")
validation_generator = test_val_datagen.flow_from_directory(
    f'{BASE_DIR}/validation',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 1.3 Test Data Loader (30 images)
print("Loading Test Data...")
test_generator = test_val_datagen.flow_from_directory(
    f'{BASE_DIR}/test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print("\nDetected Class Labels:", class_names)

# --- 2. Model Definition (Transfer Learning with MobileNetV2) ---

# 2.1 Load the Pre-trained Model (the 'smart brain')
print("\nBuilding Model with MobileNetV2 Transfer Learning...")
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,    # We discard the original top layers
    weights='imagenet'    # Load weights trained on millions of images
)

# 2.2 Freeze the Base Model
# This prevents the pre-trained weights from being destroyed by our small dataset
base_model.trainable = False 

# 2.3 Build the new classification layers (our custom 'Cat/Dog' decision-maker)
x = base_model.output
x = GlobalAveragePooling2D()(x) # Summarizes the complex features
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout to prevent overfitting
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Final Cat/Dog output

# Combine the base model and our new layers
model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Compile and Train the Model ---

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

print("\n--- Starting Training ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    # validation_steps is calculated dynamically or can be specified
    validation_steps=validation_generator.samples // BATCH_SIZE + 1 # Use +1 to cover the small validation set
)

# --- 4. Evaluate on Test Data ---

print("\n--- Final Evaluation on Test Set ---")
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")