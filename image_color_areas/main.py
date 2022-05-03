# import modules
import numpy as np
import PIL.Image



def import_image(image_path):
	img = PIL.Image.open(image_path)
	img = np.array(img)
	if img.shape[2]==3:
		img=np.append(img,np.ones((img.shape[0],img.shape[1],1),dtype=np.uint8)*255,axis=2)
	return img



def get_quantity_of_each_color(image):
	quantity_of_colors=np.zeros((grades),dtype=np.uint8)

	# for i,row in enumerate(image):
	# 	for j,pix in enumerate(row):
	# 		quantity_of_colors[pix]+=1

	quantity_of_each_color = np.bincount(image.flatten())
	return quantity_of_each_color


def rgb_to_luminance(img,grades=256):

	lum=img@np.array((0.05,0.05,0.9,0))
	lum=lum*(grades-1)/256
	return lum.round().astype(np.uint8)

def export_image(img,image_path):
	img=PIL.Image.fromarray(img,mode='RGBA')
	img.save(image_path)

grades=16
palet=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[127,0,255],[255,127,0],[255,255,127],[0,255,127],[127,255,0],[0,127,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[0,127,127],[127,0,127],[127,127,127]]


def main():

	img=import_image('___files/img1.png')
	img_lum=rgb_to_luminance(img,grades=grades)

	colors=get_quantity_of_each_color(img_lum)

	colors_temp=colors.flatten()
	colors_temp.sort()
	n_max=colors_temp[-4]
	for i in colors.shape:
		if colors[i-1]<n_max/2:
			colors[i]=None
	
	quantity_of_colors=np.count_nonzero(colors)

	img_layer=np.zeros((img.shape[0],img.shape[1],4),dtype=np.uint8)



	for i,row in enumerate(img_lum):
		for j,pix in enumerate(row):
			if colors[pix] is not None:
				img_layer[i,j]=palet[pix]+[255]
			else:
				img_layer[i,j]=(255,255,255,255)
				img_layer[i,j]=(0,0,0,255)
			pass



	export_image(img_layer,'___files/img1_layer.png')
	



if __name__ == '__main__':
	main()




