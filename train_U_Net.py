import os
from time import time

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

from make_U_Net import make_U_Net
from other_functions import *

EPOCHS = 30
BATCH_SIZE = 128

input_data, target_data = process_images(os.path.join("dataset","train","images"),os.path.join("dataset","train","masks"))
validation = process_images(os.path.join("dataset","val","images"),os.path.join("dataset","val","masks"))

model = make_U_Net()
model.summary()
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

steps = input_data.shape[0] // BATCH_SIZE
validation_steps = validation[0].shape[0] // BATCH_SIZE

tensorboard = TensorBoard(log_dir=os.path.join("logs", "{}".format(time())))
model.fit(x=input_data, y=target_data, steps_per_epoch=steps, epochs=EPOCHS, validation_data=validation, validation_steps=validation_steps, shuffle=True, callbacks=[tensorboard])

save_to_json(model, 'U-net')
