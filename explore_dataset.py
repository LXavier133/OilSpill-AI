import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NUM_SAMPLES = 1

dir = os.path.join(os.getcwd(),"dataset","train")

def show_both(img1, img2):

	img_read1 = mpimg.imread(img1)
	plt.subplot(2, 1, 1)
	plt.imshow(img_read1)
	plt.axis('off')

	img_read2 = mpimg.imread(img2)
	plt.subplot(2, 1, 2)
	plt.imshow(img_read2)
	plt.axis('off')

	plt.tight_layout()
	plt.show()

for i in range(NUM_SAMPLES):
	img=random.choice(os.listdir(os.path.join(dir,"images")))
	mask=img.replace("jpg","png")	
	show_both(os.path.join(dir,"images",img),os.path.join(dir,"masks",mask))
	
