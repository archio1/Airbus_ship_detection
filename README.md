# Airbus Ship Detection Challenge - Semantic Segmentation

This project aims to solve one of the problems from the Kaggle platform, specifically the Airbus Ship Detection Challenge. The goal is to build a semantic segmentation model to detect ships in aerial images.

## Problem Description

The problem involves developing a model that can accurately identify and segment ships in satellite images. Given a large dataset of satellite images, the task is to create a model that can predict pixel-level masks for ship instances in the images.

## Solution Overview

To tackle this challenge, the following approach has been used:

1. **Semantic Segmentation Model**: The solution utilizes the U-Net architecture, a popular choice for image segmentation tasks. U-Net is known for its ability to capture detailed information and preserve spatial context, making it suitable for this task.

2. **tf.keras**: The implementation of the model is done using `tf.keras`, a high-level deep learning API that allows for easy model development and training. It leverages the capabilities of TensorFlow while providing a user-friendly interface.

3. **Dice Score**: The evaluation metric used for the model is the Dice coefficient, which measures the similarity between the predicted masks and the ground truth masks. The Dice coefficient is a commonly used metric for semantic segmentation tasks and provides a measure of segmentation accuracy.

4. **Python**: The entire solution is implemented using the Python programming language, taking advantage of its rich ecosystem of libraries and tools for data manipulation, image processing, and deep learning.

## Files

The repository contains the following files:

- `unet.py`: Python script for training the semantic segmentation model.
- `inference.py`: Python script for evaluating the trained model on test data.

## Usage

To train and evaluate the model, follow these steps:

1. Install the required dependencies listed in `requirements.txt`.

2. Prepare the dataset by downloading the Airbus Ship Detection Challenge data from Kaggle and organizing it into appropriate directories.

3. Adjust the configuration parameters in the scripts, such as file paths, hyperparameters, and training settings, to match your setup and preferences.

4. Run `unet.py` to train the model. The script will load the data, preprocess it, train the model using the U-Net architecture, and save the trained model weights.

5. Run `inference.py` to evaluate the trained model on test data. The script will load the test data, preprocess it, load the trained model, perform inference, and compute the Dice scores for evaluation.

Feel free to explore and modify the code as per your requirements and experiment with different settings to improve the model performance.

## Conclusion

The semantic segmentation model developed using the U-Net architecture and tf.keras provides a solution to the Airbus Ship Detection Challenge. By leveraging deep learning techniques and the powerful features of the U-Net model, accurate ship segmentation can be achieved in satellite images.

The provided code and instructions enable further exploration and improvement of the model, allowing for experimentation with different architectures, hyperparameters, and data augmentation techniques.

For more details, please refer to the individual code files and comments within the codebase.