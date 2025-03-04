# Dataset Acquisition Guide

## CIFAR-10 Dataset

The CIFAR-10 dataset is a widely used benchmark for image classification. It consists of 60,000 32×32 color images in 10 different classes, with 50,000 training images and 10,000 test images.

### How to Download CIFAR-10

There are two main ways to acquire the CIFAR-10 dataset:

1. **Using Keras Datasets API:**

   The simplest method is to use TensorFlow/Keras, which provides a built-in function to load the dataset:

   ```python
   from tensorflow.keras.datasets import cifar10

   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   ```

   This function automatically downloads the dataset if it is not already present on your system.

2. **Manual Download:**

   You can manually download CIFAR-10 from the official website:
   
   - **URL:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

   After downloading, extract the files and use the provided Python scripts or libraries (such as `pickle` in Python) to load the data.

### Data Preprocessing

In our project, the CIFAR-10 images (originally 32×32 pixels) are resized to 224×224 pixels to be compatible with the Vision Transformer (ViT) model. Additionally, pixel values are normalized to the range [0, 1].

Example code snippet for preprocessing:
```python
import tensorflow as tf

def preprocess_image(image):
    # Convert to float and resize to 224x224
    image = tf.image.resize(tf.cast(image, tf.float32), (224, 224))
    # Normalize pixel values to [0, 1]
    return image / 255.0
```
