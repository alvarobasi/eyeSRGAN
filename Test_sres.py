import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Network
import utils
import os


# from custom_generator import DataGenerator


def _map_fn(image_path):
    image_high_res = tf.io.read_file(image_path)
    image_high_res = tf.image.decode_jpeg(image_high_res, channels=3)
    image_high_res = tf.image.convert_image_dtype(image_high_res, dtype=tf.float32)
    image_high_res = tf.image.random_flip_left_right(image_high_res)
    image_low_res = tf.image.resize(image_high_res, size=[21, 97])

    return image_low_res, image_high_res


if __name__ == "__main__":

    allowed_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    data_format = 'channels_last'
    tf.keras.backend.set_image_data_format(data_format)
    print("Keras: ", tf.keras.__version__)
    print("Tensorflow: ", tf.__version__)
    print("Image format: ", tf.keras.backend.image_data_format())
    utils.print_available_devices()

    batch_size = 2
    target_shape = (84, 388)
    downscale_factor = 4

    shared_axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    axis = -1 if data_format == 'channels_last' else 1

    dataset_path = './datasets/User_20/'

    # batch_gen = DataGenerator(path=dataset_path,
    #                           batch_size=batch_size,
    #                           downscale_factor=4,
    #                           target_shape=target_shape,
    #                           shuffle=True,
    #                           crop_mode='fixed_size',
    #                           color_mode='rgb',
    #                           data_format=data_format)

    if data_format == 'channels_last':
        target_shape = target_shape + (3,)
        shape = (target_shape[0] // downscale_factor, target_shape[1] // downscale_factor, 3)
    else:
        target_shape = (3,) + target_shape
        shape = (3, target_shape[1] // downscale_factor, target_shape[2] // downscale_factor)

    list_file_path = 'E:\\TFM\\outputs\\listado_imagenes_test.npy'
    if os.path.isfile(list_file_path):
        list_files = np.load(list_file_path)
    else:
        list_files = utils.get_list_of_files(dataset_path)

    np.random.shuffle(list_files)

    # Dataset creation.
    train_ds = tf.data.Dataset.from_tensor_slices(list_files).map(_map_fn,
                                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(5000)
    train_ds = train_ds.repeat(count=-1)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = train_ds.__iter__()

    model_path_mse = 'E:\\TFM\\outputs\\checkpoints\\SRResNet-MSE\\best_weights.hdf5'
    model_path_srgan = 'E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\generator_best.h5'

    # lr_images, hr_images = batch_gen.next()
    _, hr_images = next(iterator)

    generator_mse = Network.Generator(data_format=data_format, axis=axis, shared_axis=shared_axis).build()
    generator_srgan = Network.Generator(data_format=data_format, axis=axis, shared_axis=shared_axis).build()

    generator_mse.load_weights(model_path_mse)
    generator_srgan.load_weights(model_path_srgan)

    predicted_images_mse = generator_mse.predict(hr_images)
    predicted_images_srgan = generator_srgan.predict(hr_images)

    for index in range(batch_size):
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(utils.deprocess_HR(predicted_images_mse[index]).astype(np.uint8))
        ax.axis("off")
        ax.set_title("MSE")

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(utils.deprocess_LR(hr_images[index]).astype(np.uint8))
        ax.axis("off")
        ax.set_title("Original")

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(utils.deprocess_HR(predicted_images_srgan[index]).astype(np.uint8))
        ax.axis("off")
        ax.set_title("SRGAN")

        plt.show()
