# Chest Cancer Classification using VGG16

This project implements a deep learning-based classification system to identify different types of chest cancer using CT scan images. The model leverages transfer learning with the VGG16 architecture and is trained on a publicly available dataset structured into four classes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreyas074/Deep-Learning-Based-Chest-Cancer-Classification-System-Using-VGG16/blob/main/Chest_Cancer_Classification.ipynb)

---

## Dataset

Source: [Kaggle - Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

Directory structure:
```
Data/
├── train/
├── valid/
└── test/
```

Classes:
- Adenocarcinoma
- Large cell carcinoma
- Squamous cell carcinoma
- Normal

---

## Methodology

### 1. Environment Setup
- Notebook is built for Google Colab.
- Kaggle API is used to download and unzip the dataset.
- Dataset directories for train, validation, and test are defined and inspected.

### 2. Data Augmentation and Preprocessing
Used `ImageDataGenerator` with the following augmentations:
- Rotation range: 40
- Width and height shift range: 0.2
- Shear range: 0.2
- Zoom range: 0.2
- Horizontal flip: True
- Fill mode: 'nearest'
- Preprocessing function: `preprocess_input` (compatible with ResNet and VGG)

### 3. Model Architecture

Base model:
- VGG16 with `include_top=False`
- Pre-trained on ImageNet
- All layers frozen

Custom classification head:
```python
Sequential([
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])
```

---

## Training

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Batch size: 16
- Input shape: 224x224x3
- Class mode: Categorical
- Metrics: Accuracy

Data is fed using `flow_from_directory` for train, validation, and test sets.

---

## Evaluation

Model performance is evaluated on the test set using:
- Confusion matrix
- Classification report (precision, recall, f1-score)
- Accuracy and loss plots over epochs

---

## Dependencies

- Python 3.7+
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib

Install using:
```bash
pip install tensorflow keras scikit-learn matplotlib
```

---

## Results

The model demonstrates reliable performance across the four classes. Data augmentation significantly improves generalization, especially with limited training samples.

---

## Credits

- Dataset by Mohamed Hany on Kaggle
- Pre-trained weights from Keras Applications

---

## Contact

Email: shreyasbedi9@gmail.com  
LinkedIn: https://www.linkedin.com/in/shreyasbedi
