import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def make_U_Net():
	"""
	Make the U-Net CNN
	
	:return: Keras' model of the CNN
	"""

	#Going down the U
	
	input_image = Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))
	c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_1.1")(input_image)
	c12 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_1.2")(c11)
	p1  = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="max_pool_1")(c12)

	c21 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_2.1")(p1)
	c22 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_2.2")(c21)
	p2  = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="max_pool_2")(c22)

	c31 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_3.1")(p2)
	c32 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_3.2")(c31)
	p3  = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="max_pool_3")(c32)

	c41 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_4.1")(p3)
	c42 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_4.2")(c41)
	p4  = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="max_pool_4")(c42)

	c51 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_5.1")(p4)
	c52 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_5.2")(c51)

	#Going up the U

	u1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name="up_conv_1")(c52)
	u1 = concatenate([u1, c42], name="concat_1")
	c61 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_6.1")(u1)
	c62 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_6.2")(c61)
	 
	u2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name="up_conv_2")(c62)
	u2 = concatenate([u2, c32], name="concat_2")
	c71 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_7.1")(u2)
	c72 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_7.2")(c71)
	 
	u3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',name="up_conv_3")(c72)
	u3 = concatenate([u3, c22])
	c81 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_8.1")(u3)
	c82 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_8.2")(c81)
	 
	u4 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same',name="up_conv_4")(c82)
	u4 = concatenate([u4, c12], name="concat_3")
	c91 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_9.1")(u4)
	c92 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name="conv_9.2")(c91)
	 
	output = Conv2D(4, (1, 1), activation='softmax')(c92)
	 
	model = Model(inputs=[input_image], outputs=[output], name="U-Net")
	
	return model
	
model = make_U_Net()	
model.summary()
