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

from tensorflow.python.keras.layers import Input


def _map_fn(image_path):
    image_high_res = tf.io.read_file(image_path)
    image_high_res = tf.image.decode_jpeg(image_high_res, channels=3)
    image_high_res = tf.image.convert_image_dtype(image_high_res, dtype=tf.float32)
    image_high_res = tf.image.random_flip_left_right(image_high_res)
    image_low_res = tf.image.resize(image_high_res, size=[21, 97])
    image_high_res = (image_high_res - 0.5) * 2

    return image_low_res, image_high_res


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
    return 0.006 * K.mean(K.square(features_extractor(preprocess_vgg(y_pred)) -
                                   features_extractor(preprocess_vgg(y_true))),
                          axis=-1)


def build_vgg(target_shape_vgg):
    vgg19 = VGG19(include_top=False, input_shape=target_shape_vgg, weights='imagenet')

    vgg19.trainable = False
    for layer in vgg19.layers:
        layer.trainable = False

    vgg_model = Model(inputs=vgg19.input, outputs=vgg19.layers[20].output, name="VGG")

    return vgg_model


def get_gan_model(discriminator_gan, generator_gan, input_shape):
    discriminator_gan.trainable = False

    input_gan = Input(shape=input_shape, name="Entrada_GAN")
    output_generator = generator_gan(input_gan)
    output_discriminator = discriminator_gan(output_generator)

    gan_model = Model(inputs=input_gan, outputs=[output_generator, output_discriminator], name="SRGAN")
    gan_model.compile(loss=[vgg_loss, 'binary_crossentropy'], loss_weights=[1, 1e-3],
                      optimizer=common_optimizer)

    # tf.keras.utils.plot_model(gan_model, 'E:\\TFM\\outputs\\model_imgs\\gan_model.png', show_shapes=True)

    discriminator_gan.trainable = True

    return gan_model


if __name__ == "__main__":

    allowed_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    data_format = 'channels_last'
    keras.backend.set_image_data_format(data_format)
    print("Keras: ", keras.__version__)
    print("Tensorflow: ", tf.__version__)
    print("Image format: ", keras.backend.image_data_format())
    utils.print_available_devices()

    batch_size = 6
    target_shape = (84, 388)
    downscale_factor = 4

    shared_axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    axis = -1 if data_format == 'channels_last' else 1

    dataset_path = './datasets/A_guadiana_final/'

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

    list_file_path = 'E:\\TFM\\outputs\\listado_imagenes.npy'
    if os.path.isfile(list_file_path):
        list_files = np.load(list_file_path)
    else:
        list_files = utils.get_list_of_files(dataset_path)

    np.random.shuffle(list_files)

    # Dataset creation.
    train_ds = tf.data.Dataset.from_tensor_slices(list_files).map(_map_fn,
                                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(5000)
    train_ds = train_ds.repeat(count=-1)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

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

    common_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9)

    epochs = 5
    steps_per_epoch = int(len(list_files) // batch_size)

    eval_freq = 1000
    info_freq = 100
    checkpoint_freq = 2000

    if os.path.isdir('E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\'):
        shutil.rmtree('E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\')
    os.makedirs('E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\')

    if os.path.isdir('E:\\TFM\\outputs\\model_imgs\\'):
        shutil.rmtree('E:\\TFM\\outputs\\model_imgs\\')
    os.makedirs('E:\\TFM\\outputs\\model_imgs\\')

    if os.path.isdir('E:\\TFM\\outputs\\results\\'):
        shutil.rmtree('E:\\TFM\\outputs\\results\\')
    os.makedirs('E:\\TFM\\outputs\\results\\')

    discriminator = Network.Discriminator(input_shape=target_shape, axis=axis, data_format=data_format).build()
    # discriminator.load_weights('E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\discriminator.h5')
    discriminator.compile(loss='binary_crossentropy', optimizer=common_optimizer)

    generator = Network.Generator(data_format=data_format,
                                  axis=axis, shared_axis=shared_axis).build()
    generator.load_weights('E:\\TFM\\outputs\\checkpoints\\SRResNet-MSE\\best_weights.hdf5')
    # generator.load_weights('E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\generator_best.h5')

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
                                      path="E:\\TFM\\outputs\\results\\img_{}_{}_{}".format(epoch, step, index),
                                      data_format=data_format)

            # Every checkpoint_freq discriminator and generator weights are saved.
            if step % checkpoint_freq == 0 and step != 0:

                current_loss = np.mean(g_losses_mse_vgg[-info_freq::])
                generator.save_weights(
                    "E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\generator_{}_{:.4f}.h5".format(epoch, current_loss),
                    overwrite=True)

                if latest_loss > current_loss:
                    generator.save_weights(
                        "E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\generator_best.h5",
                        overwrite=True)

                    print("Model upgraded from {:4f} to {:4f}.\n".format(latest_loss,
                                                                         np.mean(g_losses_mse_vgg[-info_freq::])))
                    latest_loss = current_loss
                else:
                    print("Model not upgraded -> {:4f} is lower than {:4f}.\n".format(latest_loss,
                                                                                      np.mean(g_losses_mse_vgg[
                                                                                              -info_freq::])))

                discriminator.save_weights("E:\\TFM\\outputs\\checkpoints\\SRGAN-VGG54\\discriminator.h5",
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
