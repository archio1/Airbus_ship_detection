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

# Parameters
BATCH_SIZE = 4  # Train batch size
EDGE_CROP = 16  # While building the model
NB_EPOCHS = 5  # Training epochs
GAUSSIAN_NOISE = 0.1  # To be used in a layer in the model
UPSAMPLE_MODE = 'SIMPLE'  # SIMPLE ==> UpSampling2D, else Conv2DTranspose
NET_SCALING = None  # Downsampling inside the network
IMG_SCALING = (1, 1)  # Downsampling in preprocessing
VALID_IMG_COUNT = 400  # Valid batch size
MAX_TRAIN_STEPS = 200


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    Input arguments -
    mask_rle: Mask of one ship in the train image
    shape: Output shape of the image array
    '''
    s = mask_rle.split()  # Split the mask of each ship that is in RLE format
    starts, lengths = [np.asarray(x, dtype=int) for x in
                       (s[0:][::2], s[1:][::2])]  # Get the start pixels and lengths for which image has ship
    ends = starts + lengths - 1  # Get the end pixels where we need to stop
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)  # A 1D vec full of zeros of size = 768*768
    for lo, hi in zip(starts, ends):  # For each start to end pixels where ship exists
        img[lo:hi + 1] = 1  # Fill those values with 1 in the main 1D vector
    '''
    Returns -
    Transposed array of the mask: Contains 1s and 0s. 1 for ship and 0 for background
    '''
    return img.reshape(shape).T


def masks_as_image(in_mask_list):
    '''
    Input -
    in_mask_list: List of the masks of each ship in one whole training image
    '''
    all_masks = np.zeros((768, 768), dtype=np.int16)  # Creating 0s for the background
    for mask in in_mask_list:  # For each ship rle data in the list of mask rle
        if isinstance(mask, str):  # If the datatype is string
            all_masks += rle_decode(mask)  # Use rle_decode to create one mask for whole image
    '''
    Returns - 
    Full mask of the training image whose RLE data has been passed as an input
    '''
    return np.expand_dims(all_masks, -1)


def sample_ships(in_df, base_rep_val=1500):
    '''
    Input Args:
    in_df - dataframe we want to apply this function
    base_val - random sample of this value to be taken from the data frame
    '''
    if in_df['ships'].values[0] == 0:
        return in_df.sample(base_rep_val // 3)  # Random 1500//3 = 500 samples taken whose ship count is 0 in an image
    else:
        return in_df.sample(base_rep_val)  # Random 1500 samples taken whose ship count is not 0 in an image


def make_image_gen(in_df, batch_size=BATCH_SIZE):
    '''
    Inputs -
    in_df - data frame on which the function will be applied
    batch_size - number of training examples in one iteration
    '''
    all_batches = list(in_df.groupby('ImageId'))  # Group ImageIds and create list of that dataframe
    out_rgb = []  # Image list
    out_mask = []  # Mask list
    while True:  # Loop for every data
        np.random.shuffle(all_batches)  # Shuffling the data
        for c_img_id, c_masks in all_batches:  # For img_id and msk_rle in all_batches
            rgb_path = train_image_dir / c_img_id  # Get the img path
            c_img = imread(str(rgb_path))  # img array
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)  # Create mask of rle data for each ship in an img
            out_rgb += [c_img]  # Append the current img in the out_rgb / img list
            out_mask += [c_mask]  # Append the current mask in the out_mask / mask list
            if len(out_rgb) >= batch_size:  # If length of list is more or equal to batch size then
                yield np.stack(out_rgb) / 255.0, np.stack(
                    out_mask)  # Yeild the scaled img array (b/w 0 and 1) and mask array (0 for bg and 1 for ship)
                out_rgb, out_mask = [], []  # Empty the lists to create another batch


def create_aug_gen(in_gen, seed=None):
    '''
    Takes in -
    in_gen - train data generator, seed value
    '''
    np.random.seed(
        seed if seed is not None else np.random.choice(range(9999)))  # Randomly assign seed value if not provided
    for in_x, in_y in in_gen:  # For imgs and msks in train data generator
        seed = 12  # Seed value for imgs and msks must be same else augmentation won't be same

        # Create augmented imgs
        g_x = image_gen.flow(255 * in_x,
                             # Inverse scaling on imgs for augmentation
                             batch_size=in_x.shape[0],  # batch_size = 3
                             seed=seed,  # Seed
                             shuffle=True)  # Shuffle the data

        # Create augmented masks
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        '''Yeilds - augmented scaled imgs and msks array'''
        yield next(g_x) / 255.0, next(g_y)


# Conv2DTranspose upsampling
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


# Upsampling without Conv2DTranspose
def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


def get_model(input_img, pp_in_layer, balanced_train_df, valid_x, valid_y):
    # Upsampling method choice
    if UPSAMPLE_MODE == 'DECONV':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # If NET_SCALING is defined then do the next step else continue ahead
    if NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

    # To avoid overfitting and fastening the process of training
    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)  # Useful to mitigate overfitting
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)

    if NET_SCALING is not None:
        d = layers.UpSampling2D(NET_SCALING)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])

    seg_model.summary()
    seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef])
    # Best model weights
    weight_path = "{}_weights.best.hdf5".format('seg_model')

    # Monitor validation dice coeff and save the best model weights
    checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)

    # Reduce Learning Rate on Plateau
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                       patience=3,
                                       verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)

    # Stop training once there is no improvement seen in the model
    early = EarlyStopping(monitor="val_dice_coef",
                          mode="max",
                          patience=15)  # probably needs to be more patient, but kaggle time is limited

    # Callbacks ready
    callbacks_list = [checkpoint, early, reduceLROnPlat]

    # Finalizing steps per epoch
    step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0] // BATCH_SIZE)

    # Final augmented data being used in training
    aug_gen = create_aug_gen(make_image_gen(balanced_train_df))

    # Save loss history while training
    loss_history = [seg_model.fit_generator(aug_gen,
                                            steps_per_epoch=step_count,
                                            epochs=NB_EPOCHS,
                                            validation_data=(valid_x, valid_y),
                                            callbacks=callbacks_list,
                                            workers=1)]

    # Save the weights to load it later for test data
    seg_model.load_weights(weight_path)
    seg_model.save('seg_model.h5')


# Dice coeff
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


# Dice with BCE
def dice_p_bce(y_true, y_pred, alpha=1e-3):
    dice_loss = 1 - dice_coef(y_true, y_pred)  # Compute the Dice loss
    bce_loss = K.binary_crossentropy(y_true, y_pred)  # Compute the BCE loss
    combo_loss = dice_loss + alpha * bce_loss  # Combine the losses

    return combo_loss


def data_preprocessing(masks, train_image_dir):
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids.index += 1  # Incrimenting all the index by 1
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(
        lambda c_img_id: (train_image_dir / c_img_id).stat().st_size / 1024
    )
    unique_img_ids = unique_img_ids[unique_img_ids.file_size_kb > 35]
    masks.drop(['ships'], axis=1, inplace=True)
    masks.index += 1
    train_ids, valid_ids = train_test_split(unique_img_ids, test_size=0.3, stratify=unique_img_ids['ships'])
    # Create train data frame
    train_df = pd.merge(masks, train_ids)

    # Create test data frame
    valid_df = pd.merge(masks, valid_ids)

    train_df['grouped_ship_count'] = train_df.ships.map(lambda x: (x + 1) // 2).clip(0, 7)

    balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)

    train_gen = make_image_gen(balanced_train_df)

    valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

    return balanced_train_df, train_gen, valid_x, valid_y


def data_generator():
    dg_args = dict(rotation_range=15,  # Degree range for random rotations
                   horizontal_flip=True,  # Randomly flips the inputs horizontally
                   vertical_flip=True,  # Randomly flips the inputs vertically
                   data_format='channels_last')  # channels_last refer to (batch, height, width, channels)

    image_gen = ImageDataGenerator(**dg_args)
    label_gen = ImageDataGenerator(**dg_args)

    return image_gen, label_gen


if __name__ == "__main__":
    train_image_dir = Path('airbus-ship-detection/train_v2')
    test_image_dir = Path('airbus-ship-detection/test_v2')

    masks = pd.read_csv("airbus-ship-detection/train_ship_segmentations_v2.csv")

    balanced_train_df, train_gen, valid_x, valid_y = data_preprocessing(masks, train_image_dir)
    image_gen, label_gen = data_generator()

    # Augment the train data
    cur_gen = create_aug_gen(train_gen, seed=42)
    t_x, t_y = next(cur_gen)
    gc.collect()

    # Building the layers of UNET
    input_img = layers.Input(t_x.shape[1:], name='RGB_Input')
    pp_in_layer = input_img

    get_model(input_img, pp_in_layer, balanced_train_df, valid_x, valid_y)
