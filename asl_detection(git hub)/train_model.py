import os
import joblib
import gzip
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Ensure preprocessed data exists
if not os.path.exists('preprocessed_data.pkl.gz'):
    raise FileNotFoundError("Error: 'preprocessed_data.pkl.gz' not found! Run preprocessing first.")

# Load preprocessed data
with gzip.open("preprocessed_data.pkl.gz", "rb") as f:
    X_train, X_test, Y_train, Y_test = joblib.load(f)
print('✅ Preprocessed data loaded successfully')

# Convert class labels to numerical form
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
Y_test = label_encoder.transform(Y_test)  # Ensure it's fitted before transforming
print("✅ Labels converted to numerical format!")

# Train SVM classifier
svm_model = SVC(kernel="linear")  # Linear SVM
svm_model.fit(X_train, Y_train)
print("✅ Model training complete!")

# Predict on the test set
Y_pred = svm_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(Y_test, Y_pred)
print(f'🎯 Accuracy Score: {accuracy * 100:.2f}%')

# Print detailed classification report
print("\n📊 Classification Report:")
print(classification_report(Y_test, Y_pred))

# Save the trained model and label encoder
with gzip.open("svm_asl_model.pkl.gz", "wb") as f:
    joblib.dump(svm_model, f, compress=9)  # Adjust compression level (1-9)

with gzip.open("label_encoder.pkl.gz", "wb") as f:
    joblib.dump(label_encoder, f, compress=9)

print("✅ Training model and label encoder saved successfully!")
