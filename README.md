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

## Project Structure

The repository contains the following files:

- `exploratory_data_analysis.ipynb`: Jupyter Notebook for exploratory data analysis.
- `inference.py`: Python script for performing inference and generating predicted masks.
- `unet.py`: Python script containing the implementation of the U-Net architecture for the neural network.
- `seg_model.h5`: Trained model file in the HDF5 format.
- `seg_model_weights_best.h5`: Trained model weights file in the HDF5 format.
- `requirements.txt`: Text file listing the required dependencies for running the project.
- `airbus-ship-detection/`: Directory containing the Airbus Ship Detection dataset.

## Running the Project
To run the project, follow the instructions below:

1. First of all create an environment
2. To explore the dataset and perform data analysis, open `exploratory_data_analysis.ipynb` in Jupyter Notebook.

3. For running inference on new images, execute the following command: 

```python
python inference.py --input 'full path to file 
```

4. To train and evaluate the Unet model, execute the command: 
```python
python unet.py
```
Make sure you have the Airbus Ship Detection dataset available.

5. Please note that Google Colab I couldn't use for training the model in this project. Because I ran into a problem of limited memory for using the datasetю

## Conclusion

The semantic segmentation model developed using the U-Net architecture and tf.keras provides a solution to the Airbus Ship Detection Challenge. By leveraging deep learning techniques and the powerful features of the U-Net model, accurate ship segmentation can be achieved in satellite images.

The provided code and instructions enable further exploration and improvement of the model, allowing for experimentation with different architectures, hyperparameters, and data augmentation techniques.

For more details, please refer to the individual code files and comments within the codebase.