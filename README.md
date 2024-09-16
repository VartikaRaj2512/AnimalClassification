
# 🐾 Animal Classification Project

## 📜 Project Overview

This project involves classifying images of animals into 10 distinct categories using a Convolutional Neural Network (CNN) with Transfer Learning. The dataset, **Animal10**, contains approximately 28,000 medium-quality images sourced from Google Images and manually verified for accuracy. It includes some erroneous data to simulate real-world conditions.

## 📂 Dataset

The dataset can be accessed [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10). It comprises 10 categories of animals, with each image checked for quality and correctness. The categories include a diverse range of animals to ensure robust model training.

## 🚀 Getting Started

### 1. Loading and Exploring Data

1. **Loading Libraries**
   - Import necessary libraries such as TensorFlow, Keras, OpenCV, and more.

2. **Extracting Train Data**
   - Download and extract the dataset for use in training and validation.

3. **Shuffling Data**
   - Shuffle the input and target data to ensure optimal training performance.

### 2. Image Preprocessing

1. **Function for Resizing and Reshaping**
   - Define a function to resize images to a uniform dimension and reshape them for model compatibility.

2. **Image Resizing and Conversion**
   - Resize images and convert them into arrays suitable for model input.

3. **Viewing Preprocessed Images**
   - Visualize a few images after preprocessing to verify transformations.

4. **Re-Shuffling and Array Conversion**
   - Shuffle processed data again and convert it into arrays for training.

5. **Train-Test Split & One-Hot Encoding**
   - Split data into training and testing sets, one-hot encode categories, and normalize input images.

6. **Renaming Animals**
   - Ensure consistent naming of animal categories for easier reference.

7. **Data Augmentation**
   - Apply data augmentation techniques to enhance the diversity of training data.

8. **Preprocessing Test Data**
   - Apply the same preprocessing steps to test data.

### 3. Creating CNN Models with Transfer Learning

1. **CNN using VGG-16**
   - Implement a CNN model utilizing the VGG-16 architecture for feature extraction.

2. **CNN using ResNet50**
   - Build a CNN model using the ResNet50 architecture to leverage residual learning.

3. **Training Models**
   - Train both models using the preprocessed training data.

4. **Plotting Loss and Accuracy Curves**
   - Plot loss and accuracy curves to evaluate model performance and training progress.

### 4. Predicting Categories for Test Data

1. **Viewing Predictions**
   - Make predictions on test data and review the results to assess model accuracy.

## 📚 Explanation of Libraries

- **OpenCV (cv2)**: Used for reading images; note that OpenCV reads colors in BGR format, whereas PIL assumes RGB format.
- **TQDM**: Provides a progress bar for loops, enhancing user experience by showing real-time progress.
- **Utils Module**: Includes parameters such as `class_weight` for handling imbalanced data and `Shuffle` for consistent array shuffling.
- **Categorical**: Converts labeled data into one-hot vectors, facilitating multi-class classification.
- **Applications**: Provides pre-trained models for prediction and feature extraction.
- **Dropout**: A regularization technique to prevent overfitting by randomly dropping units during training.
- **Flatten**: Converts multi-dimensional data into a 1D array, preserving weight ordering for dense layers.
- **Dense Layer**: A fundamental layer in neural networks, providing fully connected layers for learning complex patterns.

## 📊 Results

Check the model performance and predictions in the provided results section.

## 🛠️ Installation and Dependencies

```bash
pip install tensorflow numpy opencv-python tqdm
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

