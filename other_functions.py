import numpy as np
from tensorflow.keras.models import model_from_json



def save_to_json(model, name):
	model_json = model.to_json()
	with open(name + ".json", 'w') as json_file:
		json_file.write(model_json)
	model.save_weights(model_name + ".h5")

def load_from_json(name):
	json_file = open(name+".json", 'r')
	loaded = json_file.read()
	json_file.close()
	loaded = model_from_json(loaded)
	loaded.load_weights(name + ".h5")
	return loaded


