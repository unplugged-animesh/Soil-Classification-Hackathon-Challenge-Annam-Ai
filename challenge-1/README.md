
# Soil Classification Pipeline

This repository contains a soil classification project using a modified ResNet18 CNN model.
The structure includes preprocessing, training, inference, and postprocessing steps.

## 🗂 Folder Structure

```
.
├── data/
│   └── soil_classification-2025/
│       ├── train/
│       ├── test/
│       ├── train_labels.csv
│       ├── test_ids.csv
│       └── sample_submission.csv
├── notebooks/
│   ├── training.ipynb
│   └── inference.ipynb
├── src/
│   ├── preprocessing.py
│   └── postprocessing.py
```

## ⚙️ Setup Instructions

1. Make sure all required libraries are installed:
```bash
pip install torch torchvision pandas scikit-learn matplotlib
```

2. Place the `data` directory in the root of your project.

3. Run preprocessing:
```bash
python src/preprocessing.py
```

4. Train the model:
Open `notebooks/training.ipynb` and run all cells.

5. Run inference:
Open `notebooks/inference.ipynb` and run all cells.

6. Postprocess and save submission:
```bash
python src/postprocessing.py
```

## 🧠 Model

- Model: `torchvision.models.resnet18(pretrained=True)`
- Final Layer: Modified to predict 4 soil types:
  - Alluvial
  - Black
  - Clay
  - Red

## 📦 Output

- Final submission file: `submission.csv`
- Evaluation report printed after inference

