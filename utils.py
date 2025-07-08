import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import os

def process_single_image(image_path):
	"""
	Transform a single RGB image into a 128x128x3 NumPy array with values between 0 and 1
	
	:param image_path: image path
	:type image_path: string
	:return: image as 128x128x3 NumPy array
	:rtype: NumPy array with dimension (128,128,3)
	"""
	img = Image.open(image_path)
	img_resized = img.resize((128,128), resample=Image.NEAREST)
	img_array = np.array(img_resized, dtype=np.float32)/255.0
	img_array = np.expand_dims(img_array, axis=0)
	return img_array

def array_to_image(mask_array, width, height):
	"""
	Transform a NumPy array with dimension (128,128) with integer entries between 0 and 3 into a PIL Image colored with given width and height
	
	:param mask_array: array to be transformed
	:type mask_array: NumPy array with dimension (128,128)
	:param width: width of the final image
	:type width: int
	:param height: height of the final image
	:type height: int
	:return: image with dimension (width, height)
	:rtype: PIL Image
	"""

	mask_map={ 0: (0, 0, 0), 1: (255, 0, 124), 2: (255, 204, 51), 3: (51, 221, 255) } # Using the same colors the dataset use

	img = np.zeros((128,128,3), dtype=np.uint8)
	
	for label, color in mask_map.items():
		m = (mask_array == label)
		img[m] = color

	img_pil = Image.fromarray(img)
	img_resized = img_pil.resize((width, height), Image.NEAREST) # Using NEAREST to avoid creating new colors

	return img_resized


def process_images(image_dir, mask_dir):
	"""
	Process whole images and masks directory into NumPy arrays
	Images will be 128x128x3 with values between 0 and 1
	Masks will be 128x128 with integer entries between 0 and 3	

	:param image_dir: directory of the images
	:type image_dir: string
	:param mask_dir: directory of the masks
	:type mask_dir: string
	:return: list of NumPy arrays with all images as NumPy arrays
	:rtype: list
	"""

	mask_map={ (0, 0, 0): 0, (255, 0, 124): 1, (255, 204, 51): 2, (51, 221, 255):3 } # Using the colors the dataset uses

	images = []
	masks = []
	
	image_dir = os.path.join(os.getcwd(),image_dir)
	mask_dir  = os.path.join(os.getcwd(),mask_dir)
	
	for filename in os.listdir(image_dir):
	
		image_path = os.path.join(image_dir,filename)
		mask_path  = os.path.join(mask_dir,filename.replace("jpg","png"))
	
		img = Image.open(image_path)
		mask = Image.open(mask_path)
	
		img_resized = img.resize((128,128),  resample=Image.NEAREST)
		mask_resized = mask.resize((128,128),resample=Image.NEAREST) # Using NEAREST to avoid creating new colors
		
		img_array = np.array(img_resized, dtype=np.float32)/255.0
		mask_array = np.array(mask_resized)
		
		# Going from 128x128x3 to 128x128. This is, from colors to labels
	
		true_mask = np.zeros((128,128), dtype=np.uint8)
		for color, label in mask_map.items():
			m = np.all(mask_array == color, axis = -1)
			true_mask[m] = label
		
		images.append(img_array)
		masks.append(true_mask)

	# Tranforming the lists into NumPy arrays
	images = np.array(images, dtype=np.float32)
	masks = np.array(masks, dtype=np.uint8)

	return images,masks

def save_to_json(model, name):
	"""
	Save model into a JSON file and the weights into a .weights.H5 file
	
	:param model: model
	:type model: Keras Model
	:param name: model name
	:type name: string
	"""
	
	model_json = model.to_json()
	with open(name + ".json", 'w') as json_file:
		json_file.write(model_json)
	model.save_weights(name + ".weights.h5")

def load_from_json(name):
	"""
	Load model from JSON file and the weights from a .weights.H5 file
	
	:param name: name of the file
	:type name: string
	:return: loaded model
	:rtype: Keras Model
	"""
	
	json_file = open(name+".json", 'r')
	loaded = json_file.read()
	json_file.close()
	loaded = model_from_json(loaded)
	loaded.load_weights(name + ".weights.h5")
	return loaded

