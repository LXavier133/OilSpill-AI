import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import os

def process_images(image_dir, mask_dir):
	mask_map={ (0, 0, 0): 0, (255, 0, 124): 1, (255, 204, 51): 2, (51, 221, 255):3 }
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
		mask_resized = mask.resize((128,128),resample=Image.NEAREST)
		img_array = np.array(img_resized, dtype=np.float32)/255.0
		mask_array = np.array(mask_resized)
		true_mask = np.zeros((128,128), dtype=np.uint8)
		for color, label in mask_map.items():
			m = np.all(mask_array == color, axis = -1)
			true_mask[m] = label
		images.append(img_array)
		masks.append(true_mask)
	images = np.array(images, dtype=np.float32)
	masks = np.array(masks, dtype=np.uint8)
	return images,masks

def save_to_json(model, name):
	model_json = model.to_json()
	with open(name + ".json", 'w') as json_file:
		json_file.write(model_json)
	model.save_weights(name + ".h5")

def load_from_json(name):
	json_file = open(name+".json", 'r')
	loaded = json_file.read()
	json_file.close()
	loaded = model_from_json(loaded)
	loaded.load_weights(name + ".h5")
	return loaded


