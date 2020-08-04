import os
import shutil
import platform

import numpy as np
import tensorflow as tf

import Network
import utils
import math

# from custom_generator import DataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

@tf.function
def _map_fn(image_path):
    image_high_res = tf.io.read_file(image_path)
    image_high_res = tf.image.decode_jpeg(image_high_res, channels=3)
    image_high_res = tf.image.convert_image_dtype(image_high_res, dtype=tf.float32)
    # image_high_res = tf.image.resize(image_high_res, size=[256, 256])  # FOR CELEBA DATASET
    image_high_res = tf.image.random_flip_left_right(image_high_res)
    image_low_res = tf.image.resize(image_high_res, size=[21, 97])
    # image_low_res = tf.image.resize(image_high_res, size=[64, 64]) # FRO CELEBA DATASET
    image_high_res = (image_high_res - 0.5) * 2

    return image_low_res, image_high_res


if __name__ == "__main__":

    # Activa o desactiva la compilación XLA para acelerar un poco el entrenamiento.
    tf.config.optimizer.set_jit(False)

    # Variable temporal para activar o desactivar AMP.
    amp_mode = True

    # Formatos permitidos para la base de datos de Celeba.
    allowed_formats = {'png', 'jpg', 'jpeg', 'bmp'}

    # Si no pongo esto, por alguna razón casca. Da error de cuDNN. 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Si se desea entrenar con los tensor cores, mixed_float16.
    if amp_mode:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    # Show compute policy in order to check mixed_precision capability.
    print('Compute dtype: %s' % tf.keras.mixed_precision.experimental.global_policy().compute_dtype)
    print('Variable dtype: %s' % tf.keras.mixed_precision.experimental.global_policy().variable_dtype)

    data_format = 'channels_last'
    tf.keras.backend.set_image_data_format(data_format)

    print("Keras: ", tf.keras.__version__)
    print("Tensorflow: ", tf.__version__)
    print("Image format: ", tf.keras.backend.image_data_format())
    utils.print_available_devices()

    batch_size = 16
    target_shape = (84, 388)
    # target_shape = (256, 256)
    # target_shape = (128, 128)
    downscale_factor = 4

    shared_axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    axis = -1 if data_format == 'channels_last' else 1

    dataset_path = './datasets/A_guadiana_final/'
    # dataset_path = './datasets/img_align_celeba/'

    if data_format == 'channels_last':
        target_shape = target_shape + (3,)
        shape = (target_shape[0] // downscale_factor, target_shape[1] // downscale_factor, 3)
    else:
        target_shape = (3,) + target_shape
        shape = (3, target_shape[1] // downscale_factor, target_shape[2] // downscale_factor)

    list_file_path = './outputs/listado_imagenes.npy'
    if os.path.isfile(list_file_path):
        list_files = np.load(list_file_path)
    else:
        list_files = utils.get_list_of_files(dataset_path)
        np.save(list_file_path, list_files)

    # list_files = utils.list_valid_filenames_in_directory(dataset_path, allowed_formats)

    np.random.shuffle(list_files)

    train_files = list_files[:100000]
    val_files = list_files[100001:]

    # Dataset creation.temporal
    train_ds = tf.data.Dataset.from_tensor_slices(list_files)
    train_ds = train_ds.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.repeat(count=-1)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    # Dataset creation validation
    valid_ds = tf.data.Dataset.from_tensor_slices(val_files)
    valid_ds = valid_ds.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = valid_ds.shuffle(buffer_size=1000)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    epochs = 5
    steps_per_epoch = int(len(list_files) // batch_size)

    common_optimizer = tf.keras.optimizers.Adam(lr=(1e-4)*math.sqrt(5), beta_1=0.9)

    # if os.path.isdir('./outputs/checkpoints/SRResNet-MSE/'):
    #     shutil.rmtree('./outputs/checkpoints/SRResNet-MSE/')
    # os.makedirs('./outputs/checkpoints/SRResNet-MSE/')

    generator = Network.Generator(data_format=data_format, axis=axis, shared_axis=shared_axis).build()
    # generator.load_weights('./outputs/checkpoints/SRResNet-MSE/best_weights.hdf5')
    generator.compile(loss='mse', optimizer=common_optimizer)

    checkpoint = ModelCheckpoint(
        filepath='./outputs/checkpoints/SRResNet-MSE/weights.{''epoch:02d}-{'
                 'loss:.4f}.hdf5',
        monitor='loss',
        save_weights_only=True,
        save_freq=2000,
        verbose=2)

    best_checkpoint = ModelCheckpoint(
        filepath='./outputs/checkpoints/SRResNet-MSE/best_weights.hdf5',
        monitor='loss',
        save_weights_only=True,
        save_best_only=True,
        save_freq=2000,
        verbose=2)

    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='min')

    # log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = TensorBoard(log_dir='logs/',
    #                                    histogram_freq=1,
    #                                    update_freq=500,
    #                                    profile_batch='200,300'
    #                                    )

    callbacks = [checkpoint, best_checkpoint, early_stop]

    history = generator.fit(x=train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks, validation_data=valid_ds)
