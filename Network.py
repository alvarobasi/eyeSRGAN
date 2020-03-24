import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, LeakyReLU, PReLU, UpSampling2D, \
    BatchNormalization, Dense, Flatten, add
from tensorflow.python.keras.models import Model


# Residual block
def res_block(inputs, axis, shared_axis):
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, use_bias=False)(inputs)
    x = BatchNormalization(axis=axis)(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=shared_axis)(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, use_bias=False)(x)
    x = BatchNormalization(axis=axis)(x)

    return add([x, inputs])


def up_block(x, shared_axis):
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, use_bias=False)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=shared_axis)(x)
    return x


class Generator(object):

    def __init__(self, data_format, axis, shared_axis, input_shape=None):
        self.shared_axis = shared_axis
        self.axis = axis
        self.B = 16
        self.data_format = data_format
        self.input_shape = input_shape

    def build(self):
        if self.input_shape is None:
            input_generator = Input(shape=(None, None, 3) if self.data_format == 'channels_last' else (3, None, None))
        else:
            input_generator = Input(shape=self.input_shape)

        x = Conv2D(filters=64, kernel_size=(9, 9),
                   strides=(1, 1), padding='same',
                   activation=None)(input_generator)

        x_input_res_block = PReLU(alpha_initializer='zeros',
                                  alpha_regularizer=None,
                                  alpha_constraint=None,
                                  shared_axes=self.shared_axis)(x)

        x = x_input_res_block

        # add B residual blocks
        for _ in range(self.B):
            x = res_block(x, self.axis, self.shared_axis)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, use_bias=False)(x)
        x = BatchNormalization(axis=self.axis)(x)

        # skip connection
        x = add([x, x_input_res_block])

        # two upscale blocks
        x = up_block(x, self.shared_axis)
        x = up_block(x, self.shared_axis)

        # final conv layer : activated with tanh -> pixels in [-1, 1]
        output_generator = Conv2D(3, kernel_size=(9, 9),
                                  strides=(1, 1), activation='tanh',
                                  use_bias=False, padding='same')(x)

        generator = Model(inputs=input_generator, outputs=output_generator, name="Generador")

        # tf.keras.utils.plot_model(generator, 'E:\\TFM\\outputs\\model_imgs\\generator_model.png',
        #                           show_shapes=True)

        return generator


def conv_block(x, filters, kernel_size, strides, axis):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
               activation=None, use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=axis)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, input_shape, data_format, axis):
        self.input_shape = input_shape
        self.data_format = data_format
        self.axis = axis

    def build(self):
        input_discriminator = Input(shape=self.input_shape)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), use_bias=False,
                   activation=None, padding='same')(input_discriminator)
        x = LeakyReLU(alpha=0.2)(x)

        x = conv_block(x, filters=64, kernel_size=(4, 4), strides=(2, 2), axis=self.axis)
        x = conv_block(x, filters=128, kernel_size=(3, 3), strides=(1, 1), axis=self.axis)
        x = conv_block(x, filters=128, kernel_size=(4, 4), strides=(2, 2), axis=self.axis)
        x = conv_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1), axis=self.axis)
        x = conv_block(x, filters=256, kernel_size=(4, 4), strides=(2, 2), axis=self.axis)
        x = conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1), axis=self.axis)
        x = conv_block(x, filters=512, kernel_size=(4, 4), strides=(2, 2), axis=self.axis)
        x = Flatten(data_format=self.data_format)(x)
        x = Dense(1024, activation=None)(x)
        x = LeakyReLU(alpha=0.2)(x)
        output_discriminator = Dense(1, activation='sigmoid')(x)

        discriminator_model = Model(inputs=input_discriminator, outputs=output_discriminator, name="Discriminador")

        tf.keras.utils.plot_model(discriminator_model, 'E:\\TFM\\outputs\\model_imgs\\discriminator_model.png',
                                  show_shapes=True)

        discriminator_model.summary()

        return discriminator_model
