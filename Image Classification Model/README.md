# Image Classification using Pre-trained VGG16 Model

This project demonstrates how to build a basic image classification model using a pre-trained VGG16 neural network. The model is trained on the CIFAR-10 dataset and evaluated for its performance.

## Prerequisites

- Python 3.6 or later
- TensorFlow
- Keras

You can install the necessary packages using:

```pip install tensorflow keras ```

### Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

### Steps
Setup the environment with TensorFlow/Keras
Load a pre-trained model (e.g., VGG16) and modify the top layers for your classification task
Prepare the dataset (e.g., CIFAR-10) and preprocess the images
Train the model and evaluate its performance



### License
This project is licensed under the MIT License - see the LICENSE file for details.

### How to Use
1. Copy the code into a Python file (e.g., `image_classification.py` ).
2. Run the script:

   ```python ICM.py```

The script will train the model on the CIFAR-10 dataset and print the test accuracy.