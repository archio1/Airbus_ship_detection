import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import models
from keras.utils import custom_object_scope
import argparse
from pathlib import Path
from model_unet import dice_coef


def load_model(model_path: Path) -> models.Model:
    """
    Load a pre-trained model from the specified path.

    Args:
        model_path (Path): Path to the model file.

    Returns:
        model (Model): Loaded pre-trained model.
        model (Model): Loaded pre-trained model.
    """
    with custom_object_scope({"dice_coef": dice_coef}):
        model = models.load_model(model_path)
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess the image for inference.

    Args:
        image_path (str): Path to the image for prediction.

    Returns:
        test_image (np.ndarray): Preprocessed image.
    """
    test_image = cv2.imread(image_path)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (768, 768))
    test_image = test_image.astype(np.float32) / 255.0

    test_image = np.expand_dims(test_image, axis=0)
    return test_image


def perform_inference(model: models.Model, test_image: np.ndarray) -> np.ndarray:
    """
    Perform inference on the image.

    Args:
        model (Model): Pre-trained model.
        test_image (np.ndarray): Preprocessed image.

    Returns:
        predicted_mask (np.ndarray): Predicted mask.
    """
    predicted_mask = model.predict(test_image)
    predicted_mask = (predicted_mask.squeeze() > 0.5).astype(
        np.uint8
    )  # Threshold the predicted mask
    return predicted_mask


def visualize_results(test_image: np.ndarray, predicted_mask: np.ndarray):
    """
    Visualize the results.

    Args:
        test_image (np.ndarray): Test image.
        predicted_mask (np.ndarray): Predicted mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(test_image[0])
    axes[0].set_title("Test Image")

    axes[1].imshow(predicted_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")

    plt.tight_layout()
    plt.show()

    if np.any(predicted_mask):
        print("Ship is present in the image.")
    else:
        print("No ship detected in the image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to the image for prediction")
    args = parser.parse_args()
    model_path = Path("model/seg_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found. Please ensure 'seg_model.h5' is in the correct directory."
        )

    if args.input is None:
        raise ValueError(
            "Please provide the path to the image for prediction using the '--input' argument."
        )
    elif not Path(args.input).is_file():
        raise FileNotFoundError(
            "Input image file not found. Please ensure the provided path is correct."
        )

    model = load_model(model_path)

    test_image = preprocess_image(args.input)

    predicted_mask = perform_inference(model, test_image)

    visualize_results(test_image, predicted_mask)
