from __future__ import division
import numpy as np
import six
from keras import backend as K
from keras.layers import Activation, Dense, Flatten, Input
from keras.layers.convolutional import AveragePooling2D, Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=3)(input)
    return Activation("relu")(norm)

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def resnet18_build(input_shape):
    input = Input(shape=input_shape)
    conv = Conv2D(filters=64, kernel_size=(7, 7),
                        strides=(2, 2), padding="same",
                        kernel_initializer= "he_normal",
                        kernel_regularizer=l2(1.e-4))(input)
    output = _bn_relu(conv)
    conv = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(output)

    # block1
    output = Conv2D(filters=64, kernel_size=(1, 1),
                               strides=(1,1),
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-4))(conv)                       
    output = _bn_relu(output)
    output = Conv2D(filters=64, kernel_size=(3, 3),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)
    output = _bn_relu(output)
    residual = Conv2D(filters=256, kernel_size=(1, 1),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)                    
    conv = _shortcut(conv, residual)

    for i in range(2):
        output = _bn_relu(conv)
        output = Conv2D(filters=64, kernel_size=(1, 1),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)

        output = _bn_relu(output)
        output = Conv2D(filters=64, kernel_size=(3, 3),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)
        output = _bn_relu(output)
        residual = Conv2D(filters=256, kernel_size=(1, 1),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)                    
        conv = _shortcut(conv, residual)

    # block2
    for i in range(8):
        st=(1,1)
        if i==0:
            st=(2,2)
        output = _bn_relu(conv)
        output = Conv2D(filters=128, kernel_size=(1, 1),
                                strides=st,
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)

        output = _bn_relu(output)
        output = Conv2D(filters=128, kernel_size=(3, 3),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)
        output = _bn_relu(output)
        residual = Conv2D(filters=512, kernel_size=(1, 1),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)                    
        conv = _shortcut(conv, residual)



    # block3
    for i in range(36):
        st=(1,1)
        if i==0:
            st=(2,2)
        output = _bn_relu(conv)
        output = Conv2D(filters=256, kernel_size=(1, 1),
                                strides=st,
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)

        output = _bn_relu(output)
        residual = Conv2D(filters=256, kernel_size=(3, 3),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)
        output = _bn_relu(output)
        residual = Conv2D(filters=1024, kernel_size=(1, 1),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)                    
        conv = _shortcut(conv, residual)


    # block4
    for i in range(3):
        st=(1,1)
        if i==0:
            st=(2,2)
        output = _bn_relu(conv)
        output = Conv2D(filters=512, kernel_size=(1, 1),
                                strides=st,
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)

        output = _bn_relu(output)
        residual = Conv2D(filters=512, kernel_size=(3, 3),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)
        output = _bn_relu(output)
        residual = Conv2D(filters=2048, kernel_size=(1, 1),
                                strides=(1,1),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(1e-4))(output)                    
        conv = _shortcut(conv, residual)
    # block finish
    block = _bn_relu(conv)

    block_shape = K.int_shape(block)
    pool = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(block)
    flatten = Flatten()(pool)
    predictions  = Dense(units=10, kernel_initializer="he_normal",
                      activation="softmax")(flatten)
    model = Model(inputs=input, outputs=predictions )
    return model


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')
batch_size = 32
nb_epoch = 200
img_rows, img_cols, img_channels = 32, 32, 3


# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

input_shape = (img_rows, img_cols,img_channels)
model = resnet18_build(input_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    # This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    validation_data=(X_test, Y_test),
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[lr_reducer, early_stopper, csv_logger])
