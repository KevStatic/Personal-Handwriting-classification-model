# 🖋️ Handwriting Classification Model

This repository contains a deep learning model designed to classify handwritten English digits (0–9) using Convolutional Neural Networks (CNNs). Built with TensorFlow and Keras, the model achieves high accuracy on the MNIST dataset and is adaptable for custom handwriting datasets.

---

## 📂 Project Structure
```
├── dataset/ # Custom dataset (optional)
├── saved_models/ # Trained models and character maps
├── scripts/ # Prediction and utility scripts
├── notebooks/ # Jupyter notebooks for data exploration
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # Project license
```


---

## 🚀 Features

- **Preprocessing**: Normalizes and reshapes images to 28x28 pixels.
- **Model**: CNN architecture with Conv2D, MaxPooling2D, Dropout, and Dense layers.
- **Training**: Trains on MNIST or custom datasets with accuracy metrics.
- **Prediction**: `predict.py` script for real-time digit classification.
- **Visualization**: `data_exploration.ipynb` for dataset analysis and visualization.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KevStatic/Personal-Handwriting-classification-model.git
   cd Personal-Handwriting-classification-model
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📈 Performance

- MNIST Dataset: Achieved 99.07% test accuracy after 10 epochs.
- Custom Dataset: Performance varies based on dataset quality and size.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


