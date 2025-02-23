import gdown
import os

# Google Drive file IDs (Replace with actual File IDs)
file_ids = {
    "svm_asl_model.pkl.gz": "16-h9NUWpWOeohF5eo1fiEuHaLtplsubz",
    "label_encoder.pkl.gz": "19QSVxjpbjqh-dHCfgwojCRFpWoFrxnfe",
    "preprocessed_data.pkl.gz": "1v9BD08fTyTp_jlwEG1qmh7cUeOQMd1J2"
}

# Download each file if not already present
for file_name, file_id in file_ids.items():
    if not os.path.exists(file_name):  # Avoid re-downloading
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)
        print(f"✅ {file_name} downloaded successfully!")
    else:
        print(f"🔹 {file_name} already exists. Skipping download.")

