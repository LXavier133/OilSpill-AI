import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
from PIL import Image
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

NUM_SAMPLES = 10

dir = os.path.join(os.getcwd(),"dataset","train")

def show_all(img1, img2, img_pil):
	""" 
	Display the image, expected mask and predicted mask into a plot
	
	:param img1: original image
	:type img1: string
	:param img2: expected mask image
	:type img2: string
	:param img_pil: predicted mask image
	:type img_pil: PIL Image
	"""
	
	img_read1 = mpimg.imread(img1)
	plt.subplot(3, 1, 1)
	plt.imshow(img_read1)
	plt.axis('off')
	plt.title("Image")

	img_read2 = mpimg.imread(img2)
	plt.subplot(3, 1, 2)
	plt.imshow(img_read2)
	plt.axis('off')
	plt.title("Expected mask")
	
	plt.subplot(3, 1, 3)
	plt.imshow(img_pil)
	plt.axis('off')
	plt.title("Predicted mask")
	

	plt.tight_layout()
	plt.show()


input_test, target_test = process_images(os.path.join("dataset","test","images"),os.path.join("dataset","test","masks"))

model = load_from_json('U-net')
model.summary()
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

score = model.evaluate(input_test,target_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

for i in range(NUM_SAMPLES):
	img=random.choice(os.listdir(os.path.join(dir,"images")))
	true_mask = img.replace("jpg","png")
	predict_mask_array = model.predict(process_single_image(os.path.join(dir,"images",img)))
	predict_mask_array = np.argmax(predict_mask_array[0], axis=-1).astype(np.uint8)
	with Image.open(os.path.join(dir,"images",img)) as img_pil:
		width,height = img_pil.size
	predict_mask = array_to_image(predict_mask_array, width, height)
	show_all(os.path.join(dir,"images",img),os.path.join(dir,"masks",true_mask),predict_mask)
