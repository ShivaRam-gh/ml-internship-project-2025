import os
import cv2
import joblib
import gzip
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# Set dataset path
print("✅ Script started...")
dataset_path = 'asl_alphabet_train'

# Set image size
IMG_SIZE = (64, 64)

# Lists to store features and labels
X = []  # Features (HOG descriptors)
Y = []  # Labels (class names)

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    hog_features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
    return hog_features

# Process dataset
classes = sorted(os.listdir(dataset_path))
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            print(f"🖼️ Processing image: {image_name}")

            # Preprocess image and extract HOG features
            features = preprocess_image(image_path)

            # Store features and labels
            X.append(features)
            Y.append(class_name)

# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Save compressed file
try:
    with gzip.open("preprocessed_data.pkl.gz", "wb", compresslevel=9) as f:
        joblib.dump((X_train, X_test, Y_train, Y_test), f)
    if os.path.exists("preprocessed_data.pkl.gz"):
        print("✅ preprocessed_data.pkl.gz saved successfully!")
    else:
        print("❌ ERROR: File not found after saving!")
except Exception as e:
    print(f"❌ ERROR: Failed to save file! {e}")



# Correct file existence check
if os.path.exists("preprocessed_data.pkl.gz"):
    print("✅ preprocessed_data.pkl.gz saved successfully!")
else:
    print("❌ ERROR: Failed to save preprocessed_data.pkl.gz!")

print("✅ Data preprocessing complete. Features saved for training!")
