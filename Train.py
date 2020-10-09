import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K

from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Model

import Network
import utils

import math

# import custom_generator

from tensorflow.python.keras.layers import Input


@tf.function
def _map_fn(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_high_res = tf.image.random_crop(image, [96, 96, 3])
    image_low_res = tf.image.resize(image_high_res, size=[24, 24])
    image_high_res = (image_high_res - 0.5) * 2

    return image_low_res, image_high_res


# @tf.function
# def _map_fn(image_path):
#     image_high_res = tf.io.read_file(image_path)
#     image_high_res = tf.image.decode_jpeg(image_high_res, channels=3)
#     image_high_res = tf.image.convert_image_dtype(image_high_res, dtype=tf.float32)
#     image_high_res = tf.image.random_flip_left_right(image_high_res)
#     image_low_res = tf.image.resize(image_high_res, size=[21, 97])
#     image_high_res = (image_high_res - 0.5) * 2
#
#     return image_low_res, image_high_res


# Credits to https://github.com/JGuillaumin.
def preprocess_vgg(x):
    # scale from [-1,1] to [0, 255]
    x += 1.
    x *= 127.5

    # RGB -> BGR
    if data_format == 'channels_last':
        x = x[..., ::-1]
    else:
        x = x[:, ::-1, :, :]

    # apply Imagenet preprocessing : BGR mean
    mean = [103.939, 116.778, 123.68]
    _IMAGENET_MEAN = K.constant(-np.array(mean))
    x = K.bias_add(x, K.cast(_IMAGENET_MEAN, K.dtype(x)))

    return x


def vgg_loss(y_true, y_pred):
    # return 0.006 * tf.keras.losses.mean_squared_error(features_extractor(preprocess_vgg(y_true)),
    #                                                   features_extractor(preprocess_vgg(y_pred)))
    return 0.006 * K.mean(K.square(features_extractor(preprocess_vgg(y_pred)) -
                                   features_extractor(preprocess_vgg(y_true))),
                          axis=-1)


def masked_vgg_loss(y_true, y_pred):
    mask_value = K.constant([[[-1.0, -1.0, 1.0]]])
    mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    masked = K.mean(K.square((features_extractor(preprocess_vgg(mask_true * y_pred)) -
                              features_extractor(preprocess_vgg(mask_true * y_true)))), axis=-1)

    return 0.006 * masked


def build_vgg(target_shape_vgg):
    vgg19 = VGG19(include_top=False, input_shape=target_shape_vgg, weights='imagenet')

    vgg19.trainable = False
    for layer in vgg19.layers:
        layer.trainable = False

    # vgg_model = Model(inputs=vgg19.input, outputs=vgg19.layers[20].output, name="VGG")
    vgg_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block5_conv4").output, name="VGG")

    return vgg_model


def get_gan_model(discriminator_gan, generator_gan, input_shape):
    discriminator_gan.trainable = False

    input_gan = Input(shape=input_shape, name="SRGAN_Input")
    output_generator = generator_gan(input_gan)
    output_discriminator = discriminator_gan(output_generator)

    gan_model = Model(inputs=input_gan, outputs=[output_generator, output_discriminator], name="SRGAN")
    gan_model.compile(loss=[vgg_loss, 'binary_crossentropy'], loss_weights=[1, 1e-3],
                      optimizer=common_optimizer)
    # gan_model.compile(loss=[masked_vgg_loss, 'binary_crossentropy'], loss_weights=[1, 1e-3],
    #                   optimizer=common_optimizer)

    # tf.keras.utils.plot_model(gan_model, 'E:\\TFM\\outputs\\model_imgs\\gan_model.png', show_shapes=True)

    discriminator_gan.trainable = True

    return gan_model


if __name__ == "__main__":

    # Activa o desactiva la compilación XLA para acelerar un poco el entrenamiento.
    tf.config.optimizer.set_jit(False)

    amp_mode = False

    allowed_formats = {'png', 'jpg', 'jpeg', 'bmp'}

    # Si no pongo esto, por alguna razón casca. Da error de cuDNN. 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if amp_mode:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    data_format = 'channels_last'
    keras.backend.set_image_data_format(data_format)
    print("Keras: ", keras.__version__)
    print("Tensorflow: ", tf.__version__)
    print("Image format: ", keras.backend.image_data_format())
    utils.print_available_devices()

    batch_size = 16
    target_shape = (96, 96)
    # target_shape = (84, 388)

    downscale_factor = 4

    shared_axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    axis = -1 if data_format == 'channels_last' else 1

    # dataset_path = './datasets/A_guadiana_final/'
    dataset_path = './datasets/train2017/'
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
        # list_files = utils.get_list_of_files(dataset_path)
        list_files = utils.list_valid_filenames_in_directory(dataset_path, allowed_formats)
        np.save(list_file_path, list_files)

    np.random.shuffle(list_files)

    # Dataset creation.temporal
    train_ds = tf.data.Dataset.from_tensor_slices(list_files)
    train_ds = train_ds.shuffle(buffer_size=500)
    train_ds = train_ds.repeat(count=-1)
    train_ds = train_ds.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = train_ds.__iter__()
    batch_LR, batch_HR = next(iterator)
    # batch_LR, batch_HR = batch_gen.next()

    print(batch_LR.numpy().shape) 
    print(batch_HR.numpy().shape)

    if data_format == 'channels_first':
        batch_HR = np.transpose(batch_HR, (0, 2, 3, 1))
        batch_LR = np.transpose(batch_LR, (0, 2, 3, 1))

    fig, axes = plt.subplots(4, 2, figsize=(7, 15))
    for i in range(4):
        axes[i, 0].imshow(utils.deprocess_LR(batch_LR.numpy()[i]).astype(np.uint8))
        axes[i, 1].imshow(utils.deprocess_HR(batch_HR.numpy()[i]).astype(np.uint8))

    common_optimizer = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9)
    # common_optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)

    epochs = 20
    steps_per_epoch = int(len(list_files) // batch_size)

    eval_freq = 3000
    info_freq = 100
    checkpoint_freq = 3000

    if os.path.isdir('./outputs/checkpoints/SRGAN-VGG54/'):
        shutil.rmtree('./outputs/checkpoints/SRGAN-VGG54/')
    os.makedirs('./outputs/checkpoints/SRGAN-VGG54/')

    if os.path.isdir('./outputs/model_imgs/'):
        shutil.rmtree('./outputs/model_imgs/')
    os.makedirs('./outputs/model_imgs/')

    if os.path.isdir('./outputs/results/'):
        shutil.rmtree('./outputs/results/')
    os.makedirs('./outputs/results/')

    discriminator = Network.Discriminator(input_shape=target_shape, axis=axis, data_format=data_format).build()
    discriminator.compile(loss='binary_crossentropy', optimizer=common_optimizer)

    generator = Network.Generator(data_format=data_format, axis=axis, shared_axis=shared_axis).build()
    generator.load_weights('./saved_weights/SRGAN-VGG54_real_bs16_8epochs/generator_best.h5')

    features_extractor = build_vgg(target_shape)

    # Building and compiling the GAN
    gan = get_gan_model(discriminator_gan=discriminator, generator_gan=generator, input_shape=shape)

    d_losses_fake = []
    d_losses_real = []
    g_losses_mse_vgg = []
    g_losses_cxent = []
    epochs_list = []
    latest_loss = 8000
    for epoch in range(int(epochs)):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, epochs))

        for step in range(int(steps_per_epoch)):

            # Every info_freq print training info.
            if step % info_freq == 0 and step != 0:
                print('Epoch {} : '.format(epoch))
                print("\t d_loss_fake = {:.4f} | d_loss_real = {:.4f}".format(np.mean(d_losses_fake[-info_freq::]),
                                                                              np.mean(d_losses_real[-info_freq::])))
                print("\t g_loss_mse_vgg = {:.4f}".format(np.mean(g_losses_mse_vgg[-info_freq::])))
                print("\t {:.4f} seconds per step\n".format(float(time.time() - start) / step))

            # Every eval_freq saves a batch of images and their predictions into results folder.
            if step % eval_freq == 0:
                # lr_images, hr_images = batch_gen.next()
                lr_images, hr_images = next(iterator)
                gen_hr_images = generator.predict(lr_images)

                for index, img in enumerate(gen_hr_images):
                    lr_image = lr_images[index]
                    hr_image = hr_images[index]
                    sr_image = img

                    utils.save_images(low_resolution_image=lr_image, original_image=hr_image,
                                      generated_image=sr_image,
                                      path="./outputs/results/img_{}_{}_{}".format(epoch, step, index),
                                      data_format=data_format)

            # Every checkpoint_freq discriminator and generator weights are saved.
            if step % checkpoint_freq == 0 and step != 0:

                current_loss = np.mean(g_losses_mse_vgg[-info_freq::])
                generator.save_weights(
                    "./outputs/checkpoints/SRGAN-VGG54/generator_{}_{:.4f}.h5".format(epoch, current_loss),
                    overwrite=True)

                if latest_loss > current_loss:
                    generator.save_weights(
                        "./outputs/checkpoints/SRGAN-VGG54/generator_best.h5",
                        overwrite=True)

                    print("Model upgraded from {:4f} to {:4f}.\n".format(latest_loss,
                                                                         np.mean(g_losses_mse_vgg[-info_freq::])))
                    latest_loss = current_loss
                else:
                    print("Model not upgraded -> {:4f} is lower than {:4f}.\n".format(latest_loss,
                                                                                      np.mean(g_losses_mse_vgg[
                                                                                              -info_freq::])))

                discriminator.save_weights("./outputs/checkpoints/SRGAN-VGG54/discriminator.h5",
                                           overwrite=True)
            print('Step {}/{}'.format(step, steps_per_epoch))

            discriminator.trainable = True

            # lr_images, hr_images = batch_gen.next()
            lr_images, hr_images = next(iterator)
            sr_images = generator.predict(lr_images)

            # Generate batch of fake and real labels. They are randomized in order to force the discriminator to improve
            # further.
            real_labels = np.random.uniform(0.7, 1.0, size=batch_size).astype(np.float32)
            fake_labels = np.random.uniform(0.0, 0.3, size=batch_size).astype(np.float32)

            d_loss_real = discriminator.train_on_batch(hr_images, real_labels)
            d_losses_real.append(d_loss_real)

            d_loss_fake = discriminator.train_on_batch(sr_images, fake_labels)
            d_losses_fake.append(d_loss_fake)

            discriminator.trainable = False

            # lr_images, hr_images = batch_gen.next()
            lr_images, hr_images = next(iterator)
            opposite_labels = np.ones((batch_size, 1)).astype(np.float32)

            # Train the generator network
            g_loss = gan.train_on_batch(lr_images, [hr_images, opposite_labels])
            g_losses_mse_vgg.append(g_loss[0])
