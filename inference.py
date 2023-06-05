import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import models
from keras.utils import custom_object_scope
import keras.backend as K
import argparse
from pathlib import Path


def dice_coef(y_true, y_pred, smooth = 1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def load_model(model_path):
    # Load the trained model
     with custom_object_scope({'dice_coef': dice_coef}):
        model = models.load_model(model_path)
        return model

def preprocess_image(image_path):
    # Load the test image
    test_image = cv2.imread(image_path)

    # Preprocess the test image
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (768, 768))
    test_image = test_image.astype(np.float32) / 255.0

    # Reshape the test image for model input
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

def perform_inference(model, test_image):
    # Perform inference
    predicted_mask = model.predict(test_image)
    predicted_mask = (predicted_mask.squeeze() > 0.5).astype(np.uint8)  # Threshold the predicted mask
    return predicted_mask

def visualize_results(test_image, predicted_mask):
    # Visualize the test image and the predicted mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the test image
    axes[0].imshow(test_image[0])
    axes[0].set_title("Test Image")

    # Show the predicted mask
    axes[1].imshow(predicted_mask, cmap='gray')
    axes[1].set_title("Predicted Mask")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Paths
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image for prediction")
    args = parser.parse_args()
    model_path = Path('seg_model.h5')

    # Load the model
    model = load_model(model_path)

    # Preprocess the test image
    test_image = preprocess_image(args.input)

    # Perform inference
    predicted_mask = perform_inference(model, test_image)

    # Visualize the results
    visualize_results(test_image, predicted_mask)