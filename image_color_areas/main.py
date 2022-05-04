# import modules
from email.mime import image
import logging
from logging import log
import numpy as np
import PIL.Image

logging.basicConfig(level=logging.INFO)

def import_image(image_path):
	img = PIL.Image.open(image_path)
	img = np.array(img)
	if img.shape[2]==3:
		img=np.append(img,np.ones((img.shape[0],img.shape[1],1),dtype=np.uint8)*255,axis=2)
	
	img=add_noise(img,noise_level=0.2,noise_mag=0.025)
	img=contrast_image(img,contrast=1.3,threshold=25)

	
	export_image(img,image_path+'_noise.png')

	return img


def add_noise(img,noise_level=0.1,noise_mag=0.1):
	for i,row in enumerate(img):
		for j,pix in enumerate(row):
			if np.random.random()<noise_level:
				img[i,j]=img[i,j]*(1-np.random.random()*noise_mag)
	return img


def contrast_image(img,contrast=1.0,threshold=5):
	img=img.astype(np.float32)
	img=img-128
	img=img*(contrast,contrast,contrast,1)
	img=img+128
	img=np.clip(img,0,255)
	img=additional_borderline_contrast(img,threshold)
	img=img.round()
	img=img.astype(np.uint8)
	return img


def additional_borderline_contrast(img,threshold=5):
	img=np.where(img<threshold,0,img)
	img=np.where(img>255-threshold,255,img)

	return img




def get_quantity_of_each_color(image):
	quantity_of_each_color = np.bincount(image.flatten())
	return quantity_of_each_color


def rgb_to_luminance(img,grades=256):

	lum=(img@np.array((1,1,1,0)))/3
	lum=lum*(grades-1)/256
	lum=lum.round()
	# lum=(lum.round()*(256/(grades-1))).round()
	return lum.astype(np.uint8)

def export_image(img,image_path):
	img=PIL.Image.fromarray(img,mode='RGBA')
	img.save(image_path)

grades=6
palet=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[127,0,255],[255,127,0],[255,255,127],[0,255,127],[127,255,0],[0,127,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[0,127,127],[127,0,127],[127,127,127]]
fast=0

def main():

	img=import_image('___files/img1.png')
	img_lum=rgb_to_luminance(img,grades=grades)

	colors=get_quantity_of_each_color(img_lum)

	colors_temp=colors.flatten()
	colors_temp.sort()
	n_max=colors_temp[-round(grades/4)]
	for i in range(colors.size):
		if colors[i]<n_max/4:
			colors[i]=0
	

	# quantity_of_colors=np.count_nonzero(colors)

	

	img_layer=paint_image(img, img_lum, colors)

	if fast==0:
		img_layer=replace_rare_colors_by_closest(img_lum, colors, img_layer)

	export_image(img_layer,'___files/img1_layer.png')


def replace_rare_colors_by_closest(img_lum, colors, img_layer):
	for i,row in enumerate(img_lum):
		for j,pix in enumerate(row):			
			if colors[pix]==0:
				k,l=i,j
				n,m=0,0
				direction=0 # 0:up, 1:right, 2:down, 3:left
				while True:
					try:
						if colors[img_lum[k,l]]!=0:
							img_layer[i,j]=img_layer[k,l]
							break
					except IndexError:
						match direction:
							case 0:
								direction=1
								m+=1
								k+=1
							case 1:
								direction=2
								l-=1
							case 2:
								direction=3
								k-=1
							case 3:
								direction=0
								l+=1
					match direction:
						case 0:
							if n==2*(m-1)*(2*m-1):
								direction=1
								m+=1
							k-=1
						case 1:
							if n==4*m*m-4*m+1:
								direction=2
							l+=1
						case 2:
							if n==4-m*m-2*m:
								direction=3
							k+=1
						case 3:
							if n==4*m*m:
								direction=0
							l-=1
					n+=1
	return img_layer


def paint_image(img, img_lum, colors):
	img_layer=np.zeros((img.shape[0],img.shape[1],4),dtype=np.uint8)

	for i,row in enumerate(img_lum):
		for j,pix in enumerate(row):
			if colors[pix]!=0:
				img_layer[i,j]=palet[pix]+[255]
			else:
				img_layer[i,j]=[255,255,255,255]
	
	return img_layer
	



if __name__ == '__main__':
	main()




