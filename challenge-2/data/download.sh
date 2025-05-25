#!/bin/bash

# === STEP 1: Install kaggle CLI ===
echo "Installing Kaggle CLI..."
pip install --upgrade kaggle

# === STEP 2: Set up kaggle API credentials ===
echo "Setting up Kaggle API credentials..."
mkdir -p ~/.kaggle

if [ ! -f "./kaggle.json" ]; then
    echo "ERROR: kaggle.json file not found in current directory."
    echo "Please download it from https://www.kaggle.com/account and place it in this directory."
    exit 1
fi

cp ./kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "API credentials configured."

# === STEP 3: Set competition slug and target directory ===
KAGGLE_COMPETITION="soil-classification-part-2"
TARGET_DIR="./"

# === STEP 4: Download competition dataset ===
echo "📥 Downloading competition data for: $KAGGLE_COMPETITION"
mkdir -p "$TARGET_DIR"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR"

# === STEP 5: Unzip all ZIP files ===
echo "📦 Unzipping files..."
unzip -q "$TARGET_DIR"/*.zip -d "$TARGET_DIR"

echo "✅ Download and extraction complete. Files saved to: $TARGET_DIR"
