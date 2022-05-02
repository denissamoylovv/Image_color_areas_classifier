# import modules
import numpy as np
import PIL.Image



def import_image(image_path):
	"""
	import image from path
	"""
	img = PIL.Image.open(image_path)
	img = np.array(img)
	return img





def get_quantity_of_each_color(image):
	"""
	get quantity of each color in image
	"""
	quantity_of_each_color = np.bincount(image.flatten())
	return quantity_of_each_color



img=import_image('___files/3.png')

img=img@np.array((0.2126,0.7152,0.0722))
img=img.round().astype(int)

colors=get_quantity_of_each_color(img)

# split array by 0 ellements
colors=np.split(colors,np.where(colors==0)[0])

print(f"{colors}")







# image_gs = np.zeros((image.shape[0], image.shape[1]))

# # image_gs[i][j]=pix

# # # numpy array to image
# image = PIL.Image.fromarray(image_gs)

# # convert to rgb 
# image = image.convert('RGB')

# # save image
# image.save('_files/3_hsv.png')




