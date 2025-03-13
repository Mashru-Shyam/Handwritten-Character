# Handwritten-Character-Classification

## Overview

This project implements and compares three different machine learning algorithms for handwritten character classification on the **MNIST** dataset. The algorithms compared are:

1. **Artificial Neural Network (ANN)**
2. **Convolutional Neural Network (CNN)**
3. **K-Nearest Neighbors (KNN)**

The goal is to find the best performing model by comparing their accuracy on the test data.

## Dataset

The **MNIST** dataset consists of 70,000 images of handwritten digits (0-9), each with a size of 28x28 pixels. The dataset is divided into:

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

Each image is a grayscale image and is labeled with the digit it represents.

You can download the dataset from the official [MNIST page](http://yann.lecun.com/exdb/mnist/) or load it directly using popular libraries such as TensorFlow or Keras.

## Algorithms Implemented

### 1. Artificial Neural Network (ANN)

- **Architecture**: A fully connected neural network with multiple hidden layers.
- **Input Layer**: 784 neurons (28x28 pixels flattened into a vector).
- **Hidden Layers**: 1 or more dense layers with ReLU activation.
- **Output Layer**: 10 neurons (one for each class, representing digits 0-9), with Softmax activation.
- **Optimizer**: Adam.
- **Loss Function**: Categorical Crossentropy.

### 2. Convolutional Neural Network (CNN)

- **Architecture**: Includes convolutional layers, max pooling layers, and fully connected layers.
- **Input Layer**: 28x28x1 (for grayscale images).
- **Convolutional Layers**: Several layers with ReLU activation, followed by max pooling.
- **Fully Connected Layers**: After convolutional layers, flattening is applied, followed by dense layers.
- **Dropout**: Used to reduce overfitting.
- **Output Layer**: 10 neurons (Softmax activation).
- **Optimizer**: Adam.
- **Loss Function**: Categorical Crossentropy.

### 3. K-Nearest Neighbors (KNN)

- **Preprocessing**: Flatten the 28x28 images into 784-dimensional vectors.
- **Distance Metric**: Euclidean distance.
- **K-value**: Tuned for optimal performance.
- **Classification**: Majority voting among the K nearest neighbors.

## Installation

1. **Clone the repository**:

   git clone https://github.com/Shyam-Mashru/Handwritten-Character-Classification.git
   cd handwritten-character-classification

## Results

After training and testing the models, the following results were obtained (example results):

- **ANN Accuracy**: 95%
- **CNN Accuracy**: 99%
- **KNN Accuracy**: 81%

## Conslusion

- CNN is better than ANN and KNN for digit recognition (OCR)
