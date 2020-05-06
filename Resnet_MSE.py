import os
import shutil

import numpy as np
import tensorflow as tf

import Network
import utils

# from custom_generator import DataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras.mixed_precision import experimental as mixed_precision


def _map_fn(image_path):
    image_high_res = tf.io.read_file(image_path)
    image_high_res = tf.image.decode_png(image_high_res, channels=3)
    image_high_res = tf.image.convert_image_dtype(image_high_res, dtype=tf.float32)
    # image_high_res = tf.image.random_flip_left_right(image_high_res)
    image_low_res = tf.image.resize(image_high_res, size=[21, 97])
    image_high_res = (image_high_res - 0.5) * 2

    return image_low_res, image_high_res


if __name__ == "__main__":

    # tf.keras.backend.clear_session()
    # tf.config.optimizer.set_jit(True)
    
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    # Show compute policy in order to check mixed_precision capability.
    print('Compute dtype: %s' % tf.keras.mixed_precision.experimental.global_policy().compute_dtype)
    print('Variable dtype: %s' % tf.keras.mixed_precision.experimental.global_policy().variable_dtype)

    data_format = 'channels_last'
    tf.keras.backend.set_image_data_format(data_format)

    print("Keras: ", tf.keras.__version__)
    print("Tensorflow: ", tf.__version__)
    print("Image format: ", tf.keras.backend.image_data_format())
    utils.print_available_devices()

    batch_size = 32
    target_shape = (84, 388)
    downscale_factor = 4

    shared_axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    axis = -1 if data_format == 'channels_last' else 1

    # dataset_path = './datasets/A_guadiana_final/'
    dataset_path = '/media/alvaro/DATA/Users/Alvaro/Documents/TFM/Project/eyesrgan/datasets/A_guadiana_final/'

    if data_format == 'channels_last':
        target_shape = target_shape + (3,)
        shape = (target_shape[0] // downscale_factor, target_shape[1] // downscale_factor, 3)
    else:
        target_shape = (3,) + target_shape
        shape = (3, target_shape[1] // downscale_factor, target_shape[2] // downscale_factor)

    # list_file_path = 'E:\\TFM\\outputs\\listado_imagenes.npy'
    list_file_path = '/media/alvaro/Alvaro/TFM/outputs/listado_imagenes.npy'
    if os.path.isfile(list_file_path):
        list_files = np.load(list_file_path)
    else:
        list_files = utils.get_list_of_files(dataset_path)
        np.save(list_file_path, list_files)

    np.random.shuffle(list_files)

    # Dataset creation.
    train_ds = tf.data.Dataset.from_tensor_slices(list_files).map(_map_fn,
                                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.shuffle(buffer_size=50)
    train_ds = train_ds.repeat(count=-1)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # num_steps = 1e5
    # steps_per_epoch = 5000
    epochs = 3
    steps_per_epoch = int(len(list_files) // batch_size)
    # epochs = int(num_steps // steps_per_epoch)

    common_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9)
    # common_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(common_optimizer)

    if os.path.isdir('/media/alvaro/Alvaro/TFM/outputs/checkpoints/SRResNet-MSE/'):
        shutil.rmtree('/media/alvaro/Alvaro/TFM/outputs/checkpoints/SRResNet-MSE/')
    os.makedirs('/media/alvaro/Alvaro/TFM/outputs/checkpoints/SRResNet-MSE/')

    generator = Network.Generator(data_format=data_format, axis=axis, shared_axis=shared_axis).build()
    # generator.load_weights('E:\\TFM\\outputs\\checkpoints\\SRResNet-MSE\\best_weights.hdf5')
    generator.compile(loss='mse', optimizer=common_optimizer)

    checkpoint = ModelCheckpoint(
        filepath='/media/alvaro/Alvaro/TFM/outputs/checkpoints/SRResNet-MSE/weights.{''epoch:02d}-{'
                 'loss:.4f}.hdf5',
        monitor='loss',
        save_weights_only=True,
        save_freq=2000,
        verbose=2)

    best_checkpoint = ModelCheckpoint(
        filepath='/media/alvaro/Alvaro/TFM/outputs/checkpoints/SRResNet-MSE/est_weights.hdf5',
        monitor='loss',
        save_weights_only=True,
        save_best_only=True,
        save_freq=2000,
        verbose=2)

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=2, verbose=1, mode='min')

    # log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = TensorBoard(log_dir=log_dir,
    #                                    histogram_freq=1,
    #                                    update_freq=300,
    #                                    profile_batch='500,700'
    #                                    )

    callbacks = [checkpoint, best_checkpoint, early_stop]

    history = generator.fit(x=train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)
