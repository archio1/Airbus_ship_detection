import numpy as np
import pandas as pd
import keras.backend as K

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.io import imread
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
from pathlib import Path
import gc

gc.enable()

BATCH_SIZE = 4              # Train batch size
EDGE_CROP = 16              # While building the model
NB_EPOCHS = 5               # Training epochs
GAUSSIAN_NOISE = 0.1        # To be used in a layer in the model
UPSAMPLE_MODE = "SIMPLE"    # SIMPLE ==> UpSampling2D, else Conv2DTranspose
NET_SCALING = None          # Downsampling inside the network
IMG_SCALING = (1, 1)        # Downsampling in preprocessing
VALID_IMG_COUNT = 400       # Valid batch size
MAX_TRAIN_STEPS = 200


def rle_decode(mask_rle: str, shape: tuple = (768, 768)) -> np.ndarray:
    """
    Decode the RLE-encoded mask to obtain the image mask.

    Args:
        mask_rle (str): Mask of one ship in the train image in RLE format.
        shape (tuple): Output shape of the image array.

    Returns:
        np.ndarray: Transposed array of the mask, containing 1s and 0s. 1 for ship and 0 for background.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    ends = starts + lengths - 1
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):  # For each start to end pixels where ship exists
        img[lo : hi + 1] = 1  # Fill those values with 1 in the main 1D vector

    return img.reshape(shape).T


def masks_as_image(in_mask_list: list) -> np.ndarray:
    """
    Convert a list of masks into a single image mask.

    Args:
        in_mask_list (list): List of the masks of each ship in one whole training image.

    Returns:
        np.ndarray: Full mask of the training image whose RLE data has been passed as an input.

    """
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)

    return np.expand_dims(all_masks, -1)


def sample_ships(in_df: pd.DataFrame, base_rep_val: int = 1500) -> pd.DataFrame:
    """
    Sample ships from the input DataFrame.

    Args:
        in_df (pd.DataFrame): DataFrame from which the ships are to be sampled.
        base_rep_val (int): Number of samples to be taken.

    Returns:
        pd.DataFrame: Sampled ships DataFrame.

    """
    if in_df["ships"].values[0] == 0:
        return in_df.sample(base_rep_val // 3)
    else:
        return in_df.sample(base_rep_val)


def make_image_gen(in_df: pd.DataFrame, batch_size: int = BATCH_SIZE):
    """
    Generate batches of training examples.

    Args:
        in_df (pd.DataFrame): DataFrame on which the function will be applied.
        batch_size (int): Number of training examples in one iteration.

    Yields:
        tuple: Tuple containing the batch of RGB images and corresponding masks.

    """
    all_batches = list(in_df.groupby("ImageId"))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = train_image_dir / c_img_id
            c_img = imread(str(rgb_path))
            c_mask = masks_as_image(c_masks["EncodedPixels"].values)
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb) / 255.0, np.stack(out_mask)
                out_rgb, out_mask = [], []


def create_aug_gen(in_gen: tuple, seed: int = None) -> tuple:
    """
    Create an augmented data generator.

    Args:
        in_gen: Training data generator.
        seed (int): Seed value for reproducibility.

    Yields:
        tuple: Tuple containing augmented batch of input images and labels.

    """
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = 12

        g_x = image_gen.flow(
            255 * in_x, batch_size=in_x.shape[0], seed=seed, shuffle=True
        )

        g_y = label_gen.flow(in_y, batch_size=in_x.shape[0], seed=seed, shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


def upsample_conv(
    filters: int, kernel_size: int, strides: tuple, padding: str
) -> layers.Conv2DTranspose:
    """
    Upsample the input using a transposed convolutional layer.
    """
    return layers.Conv2DTranspose(
        filters, kernel_size, strides=strides, padding=padding
    )


def upsample_simple(strides: tuple) -> layers.UpSampling2D:
    """
    Upsample the input using a simple upsampling layer.
    """
    return layers.UpSampling2D(strides)


def get_model(
    t_x: np.ndarray,
    balanced_train_df: pd.DataFrame,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
):
    """
    Build and train the segmentation model.

    Args:
        t_x (np.ndarray): Input tensor shape.
        balanced_train_df(pd.DataFrame): Dataframe containing balanced training data.
        valid_x (np.ndarray): Validation input tensor.
        valid_y (np.ndarray): Validation target tensor.

    Returns:
        Saved model and weights
    """
    input_img = layers.Input(t_x.shape[1:], name="RGB_Input")
    pp_in_layer = input_img
    if UPSAMPLE_MODE == "DECONV":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    if NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p4)
    c5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(c9)

    d = layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)
    d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)

    if NET_SCALING is not None:
        d = layers.UpSampling2D(NET_SCALING)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])

    seg_model.summary()
    seg_model.compile(
        optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef]
    )
    weight_path = f"model/{seg_model}_weights.best.hdf5"

    checkpoint = ModelCheckpoint(
        weight_path,
        monitor="val_dice_coef",
        verbose=1,
        save_best_only=True,
        mode="max",
        save_weights_only=True,
    )

    reduceLROnPlat = ReduceLROnPlateau(
        monitor="val_dice_coef",
        factor=0.5,
        patience=3,
        verbose=1,
        mode="max",
        epsilon=0.0001,
        cooldown=2,
        min_lr=1e-6,
    )

    early = EarlyStopping(
        monitor="val_dice_coef", mode="max", patience=15
    )  # probably needs to be more patient, but kaggle time is limited

    callbacks_list = [checkpoint, early, reduceLROnPlat]

    step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0] // BATCH_SIZE)

    aug_gen = create_aug_gen(make_image_gen(balanced_train_df))

    loss_history = [
        seg_model.fit_generator(
            aug_gen,
            steps_per_epoch=step_count,
            epochs=NB_EPOCHS,
            validation_data=(valid_x, valid_y),
            callbacks=callbacks_list,
            workers=1,
        )
    ]

    seg_model.load_weights(weight_path)
    seg_model.save("seg_model.h5")


def dice_coef(y_true, y_pred, smooth: float = 1.0):
    """
    Compute the Dice coefficient.

    Args:
        y_true (K.Tensor): True target tensor.
        y_pred (K.Tensor): Predicted target tensor.
        smooth (float): Smoothing factor.

    Returns:
        K.Tensor: Dice coefficient.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(y_true, y_pred, alpha: float = 1e-3):
    """
    Compute the Dice loss and the BCE loss.

    Args:
        y_true (K.Tensor): True target tensor.
        y_pred (K.Tensor): Predicted target tensor.
        alpha (float): Weighting factor for the BCE loss.

    Returns:
        K.Tensor: Dice loss.
    """
    dice_loss = 1 - dice_coef(y_true, y_pred)  # Compute the Dice loss
    bce_loss = K.binary_crossentropy(y_true, y_pred)  # Compute the BCE loss
    combo_loss = dice_loss + alpha * bce_loss  # Combine the losses

    return combo_loss


def data_preprocessing(masks: pd.DataFrame, train_image_dir: Path):
    """
    Perform data preprocessing and split the data into train and validation sets.

    Args:
        masks (pd.DataFrame): DataFrame containing the masks and encoded pixels.
        train_image_dir (Path): Directory containing the training images.

    Returns:
        balanced_train_df (pd.DataFrame): Balanced train data frame.
        train_gen: Train data generator.
        valid_x (np.ndarray): Validation input data.
        valid_y (np.ndarray): Validation target data.

    """
    masks["ships"] = masks["EncodedPixels"].map(
        lambda c_row: 1 if isinstance(c_row, str) else 0
    )

    unique_img_ids = masks.groupby("ImageId").agg({"ships": "sum"}).reset_index()
    unique_img_ids.index += 1  # Incrimenting all the index by 1
    unique_img_ids["has_ship"] = unique_img_ids["ships"].map(
        lambda x: 1.0 if x > 0 else 0.0
    )
    unique_img_ids["file_size_kb"] = unique_img_ids["ImageId"].map(
        lambda c_img_id: (train_image_dir / c_img_id).stat().st_size / 1024
    )
    unique_img_ids = unique_img_ids[unique_img_ids.file_size_kb > 35]
    masks.drop(["ships"], axis=1, inplace=True)
    masks.index += 1
    train_ids, valid_ids = train_test_split(
        unique_img_ids, test_size=0.3, stratify=unique_img_ids["ships"]
    )
    # Create train data frame
    train_df = pd.merge(masks, train_ids)

    # Create test data frame
    valid_df = pd.merge(masks, valid_ids)

    train_df["grouped_ship_count"] = train_df.ships.map(lambda x: (x + 1) // 2).clip(
        0, 7
    )

    balanced_train_df = train_df.groupby("grouped_ship_count").apply(sample_ships)

    train_gen = make_image_gen(balanced_train_df)

    valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

    return balanced_train_df, train_gen, valid_x, valid_y


def data_generator():
    """
    Create image and label generators for data augmentation.

    Returns:
        image_gen (ImageDataGenerator): Image generator for augmentation.
        label_gen (ImageDataGenerator): Label generator for augmentation.
    """
    dg_args = dict(
        rotation_range=15,  # Degree range for random rotations
        horizontal_flip=True,  # Randomly flips the inputs horizontally
        vertical_flip=True,  # Randomly flips the inputs vertically
        data_format="channels_last",
    )  # channels_last refer to (batch, height, width, channels)

    image_gen = ImageDataGenerator(**dg_args)
    label_gen = ImageDataGenerator(**dg_args)

    return image_gen, label_gen


if __name__ == "__main__":
    train_image_dir = Path("airbus-ship-detection/train_v2")

    masks = pd.read_csv("airbus-ship-detection/train_ship_segmentations_v2.csv")

    balanced_train_df, train_gen, valid_x, valid_y = data_preprocessing(
        masks, train_image_dir
    )
    image_gen, label_gen = data_generator()

    # Augment the train data
    cur_gen = create_aug_gen(train_gen, seed=42)
    t_x, t_y = next(cur_gen)
    gc.collect()

    get_model(t_x, balanced_train_df, valid_x, valid_y)
