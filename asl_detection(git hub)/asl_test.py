import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
import gzip

# Load trained model and label encoder
with gzip.open("svm_asl_model.pkl.gz", "rb") as f:
    svm_model = joblib.load(f)

with gzip.open("label_encoder.pkl.gz", "rb") as f:
    label_encoder = joblib.load(f)

# Set the path for test images
test_images_path = "asl_alphabet_test"

IMG_SIZE = (64, 64)  # Image size


# Function to preprocess images (same as used in training)
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Skipping {image_path}: Unable to read image")
        return None  # Skip unreadable images

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)

    # Extract HOG features
    hog_features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)

    return hog_features


# Ensure test folder exists
if not os.path.exists(test_images_path):
    raise FileNotFoundError(f"❌ Error: Test folder '{test_images_path}' not found!")

# Process and classify test images
for image_name in sorted(os.listdir(test_images_path)):  # Sort for consistent results
    image_path = os.path.join(test_images_path, image_name)

    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
        print(f"⚠️ Skipping {image_name}: Not an image file")
        continue

    features = preprocess_image(image_path)
    if features is None:
        continue  # Skip if preprocessing failed

    features = features.reshape(1, -1)  # Reshape for SVM
    prediction = svm_model.predict(features)  # Predict label index
    predicted_label = label_encoder.inverse_transform(prediction)[0]  # Convert to class name

    print(f"✅ Image: {image_name} → Predicted Sign: {predicted_label}")

    # # Show image (optional)
    # img = cv2.imread(image_path)
    # cv2.imshow(f"Predicted: {predicted_label}", img)
    # cv2.waitKey(1000)  # Show each image for 1 second
    # cv2.destroyAllWindows()
