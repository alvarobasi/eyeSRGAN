import os
import random

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tensorflow as tf
from skimage import io

import Network
import utils


# from custom_generator import DataGenerator


def _map_fn(image_path):
    image_high_res = tf.io.read_file(image_path)
    image_high_res = tf.image.decode_jpeg(image_high_res, channels=3)
    image_high_res = tf.image.convert_image_dtype(image_high_res, dtype=tf.float32)
    image_low_res = tf.image.resize(image_high_res, size=[21, 97])

    return image_high_res, image_low_res


if __name__ == "__main__":

    data_format = 'channels_last'
    allowed_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    tf.keras.backend.set_image_data_format(data_format)
    print("Keras: ", tf.keras.__version__)
    print("Tensorflow: ", tf.__version__)
    print("Image format: ", tf.keras.backend.image_data_format())
    utils.print_available_devices()

    batch_size = 1
    target_shape = (84, 388)
    downscale_factor = 4

    shared_axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    axis = -1 if data_format == 'channels_last' else 1

    dataset_path = './datasets/A_guadiana_final/'
    # dataset_path = './datasets/I2Head_f_recortada/'
    # dataset_path = './datasets/train2017/'
    # batch_gen = DataGenerator(path=dataset_path,
    #                           batch_size=batch_size,
    #                           downscale_factor=4,
    #                           target_shape=target_shape,
    #                           shuffle=True,
    #                           crop_mode='fixed_size',
    #                           color_mode='rgb',
    #                           data_format=data_format)

    # if data_format == 'channels_last':
    #     target_shape = target_shape + (3,)
    #     shape = (target_shape[0] // downscale_factor, target_shape[1] // downscale_factor, 3)
    # else:
    #     target_shape = (3,) + target_shape
    #     shape = (3, target_shape[1] // downscale_factor, target_shape[2] // downscale_factor)

    list_files = utils.get_list_of_files(dataset_path)
    # list_files = utils.list_valid_filenames_in_directory(dataset_path, allowed_formats)
    # random_list = random.sample(list_files, 1)

    # Dataset creation.
    train_ds = tf.data.Dataset.from_tensor_slices(list_files).map(_map_fn,
                                                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = train_ds.__iter__()

    # model_path_srgan = 'E:\\TFM\\outputs\\Resultados_Test\\Sin_Textura_4\\checkpoints\\SRGAN-VGG54\\generator_best.h5'
    model_path_srgan = 'saved_weights/SRGAN-VGG54/generator_best.h5'
    model_path_mse = 'saved_weights/SRResNet-MSE/best_weights.hdf5'
    # model_path_srgan = 'saved_weights/SRGAN-VGG54-20bs-2e-u18/generator_best.h5'

    generator_srgan = Network.Generator(data_format=data_format, axis=axis, shared_axis=shared_axis).build()

    generator_srgan.load_weights(model_path_mse)

    for i, file in enumerate(list_files):
        print(i)
        hr_images, lr_images = next(iterator)

        predicted_images = generator_srgan.predict(lr_images)

        base_name = os.path.splitext(file)[0]
        output_path = f'{base_name}_mse.png'
        io.imsave(output_path, utils.deprocess_HR(predicted_images[0]).astype(np.uint8))
        # io.imsave('./outputs/' + f'{i}_gen_sr.png', utils.deprocess_HR(predicted_images[0]).astype(np.uint8))
        # io.imsave('./outputs/' + f'{i}_orig.png', utils.deprocess_LR(crop_hr[0, :, :]).astype(np.uint8))
