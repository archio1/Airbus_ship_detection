# Airbus Ship Detection Challenge - Semantic Segmentation

This project aims to solve one of the problems from the Kaggle platform, specifically the Airbus Ship Detection Challenge. The goal is to build a semantic segmentation model to detect ships in aerial images.

## Problem Description

The problem involves developing a model that can accurately identify and segment ships in satellite images. Given a large dataset of satellite images, the task is to create a model that can predict pixel-level masks for ship instances in the images.

## Solution Overview

To tackle this challenge, the following approach has been used:

1. **Python**: The entire solution is implemented using the Python programming language, taking advantage of its rich ecosystem of libraries and tools for data manipulation, image processing, and deep learning.

2. **tf.keras**: The implementation of the model is done using `tf.keras`, a high-level deep learning API that allows for easy model development and training. It leverages the capabilities of TensorFlow while providing a user-friendly interface.

3. **Dice Score**: The evaluation metric used for the model is the Dice coefficient, which measures the similarity between the predicted masks and the ground truth masks. The Dice coefficient is a commonly used metric for semantic segmentation tasks and provides a measure of segmentation accuracy.

4. **Semantic Segmentation Model**: The solution utilizes the U-Net architecture, a popular choice for image segmentation tasks. U-Net is known for its ability to capture detailed information and preserve spatial context, making it suitable for this task.

## Project Structure

The repository contains the following files:

- `exploratory_data_analysis.ipynb`: Jupyter Notebook for exploratory data analysis.
- `inference.py`: Python script for performing inference and generating predicted masks.
- `unet.py`: Python script containing the implementation of the U-Net architecture for the neural network.
- `model/`: Trained model and weights file in the HDF5 format.
- `requirements.txt`: Text file listing the required dependencies for running the project.
- `airbus-ship-detection/`: Directory containing the Airbus Ship Detection dataset.

## Running the Project
To run the project, follow the instructions below:

0. kaggle competitions download -c airbus-ship-detection
1. Create a virtual environment and install dependencies from requirements.txt
2. To explore the dataset and perform data analysis, open `exploratory_data_analysis.ipynb` in Jupyter Notebook.

3. To train and evaluate the Unet model, execute the command: 
```python
python unet.py
```
Make sure you have the Airbus Ship Detection dataset available.

4. For running inference on new images, execute the following command: 

```python
python inference.py --input  C:\Users\User\PycharmProjects\pythonProject\Project_check\Airbus_ship_detection\airbus-ship-detection\test_v2\00a3ab3cc.jpg
```

## Conclusion

The semantic segmentation model developed using the U-Net architecture and tf.keras provides a solution to the Airbus Ship Detection Challenge. By leveraging deep learning techniques and the powerful features of the U-Net model, accurate ship segmentation can be achieved in satellite images.

Please note that Google Colab I couldn't use for training the model in this project. Because I ran into a problem of limited memory for using the dataset.
Model shows low accuracy. To improve the accuracy of model: increase the model's capacity, adjust the learning rate, try different loss functions, regularize the model, hyperparameter tuning.
This project was created by Artem Tatarchuk. For any questions or additional information, please contact arti123t@gmail.com.