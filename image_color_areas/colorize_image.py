# import modules
from tkinter.messagebox import NO
import numpy as np
import PIL.Image
import base64
import io


# with open('___files/img_input.png','rb') as f:
# 	print(base64.b64encode(f.read()).decode('utf-8'))



# convert numpy array image to base64 string
def image_to_base64(img:np.ndarray) -> str:


	image=PIL.Image.fromarray(img,mode='RGBA')

	b = io.BytesIO()
	image.save(b, 'png')
	img_bytes = b.getvalue()
	img_str=base64.b64encode(img_bytes).decode('utf-8')
	return img_str


def import_image_base64(img_base64_str:str) -> np.ndarray:
	img_base64_str=img_base64_str[22:]+'===='
	img_bytes=base64.b64decode(img_base64_str)
	img=import_image(io.BytesIO(img_bytes))
	return img


def import_image(image:str|io.BytesIO) -> np.ndarray:
	img = PIL.Image.open(image)
	
	img = np.array(img)
	if img.shape[2]==3:
		img=np.append(img,np.ones((img.shape[0],img.shape[1],1),dtype=np.uint8)*255,axis=2)

	
	# export_image(img,image_path+'_noise.png')

	return img

def image_preprocessing(img:np.ndarray,threshold:int=5,contrast:float = 1.0,noise_level:float=0.2,noise_mag:float=0.025) -> np.ndarray:
	img=contrast_image(img,contrast=contrast,threshold=threshold)
	img=add_noise(img,noise_level=noise_level,noise_mag=noise_mag)
	img=compressor(img)
	return img


def add_noise(img:np.ndarray,noise_level=0.1,noise_mag=0.1)->np.ndarray:
	for i,row in enumerate(img):
		for j,pix in enumerate(row):
			if np.random.random()<noise_level:
				img[i,j]=img[i,j]*(1-np.random.random()*noise_mag)
	return img


def contrast_image(img:np.ndarray,contrast=1.0,threshold=5) -> np.ndarray:
	img=img.astype(np.float32)
	img=img-128
	img=img*(contrast,contrast,contrast,1)
	img=img+128
	img=np.clip(img,0,255)
	img=additional_borderline_contrast(img,threshold)
	img=img.round()
	img=img.astype(np.uint8)
	return img


def additional_borderline_contrast(img:np.ndarray,threshold=5) -> np.ndarray:
	img=np.where(img<threshold,0,img)
	img=np.where(img>255-threshold,255,img)

	return img

def compressor(img:np.ndarray) -> np.ndarray:
	img=img.astype(np.float32)
	img=img-(np.min(img)+np.max(img))/2
	img=img/np.max(img)*127+127
	img=img.round()
	img=img.astype(np.uint8)
	return img




def get_quantity_of_each_color(image:np.ndarray) -> np.ndarray:
	quantity_of_each_color = np.bincount(image.flatten())
	return quantity_of_each_color


def rgb_to_luminance(img:np.ndarray,grades=256,clip:bool=False) -> np.ndarray:

	lum=(img@np.array((1,1,1,0)))/3
	if clip:
		lum=lum*(grades-1)/256
	lum=lum.round()
	# lum=(lum.round()*(256/(grades-1))).round()
	return lum.astype(np.uint8)

def export_image(img:np.ndarray,image_path:str) -> None:
	image=PIL.Image.fromarray(img,mode='RGBA')
	image.save(image_path)

palet=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[127,0,255],[255,127,0],[255,255,127],[0,255,127],[127,255,0],[0,127,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[0,127,127],[127,0,127],[127,127,127]]




def main():
	img=import_image('___files/img1.png')
	colored_img=create_colored_image(img,grades=4,kmean=True,palet=palet, threshold=5,contrast=1.0,noise_level=0.2,noise_mag=0.025,opacity=1)
	export_image(colored_img,'___files/img1_colored.png')




def colorize_image_to_base64(image_str:str,grades:int,kmean=False,palet=palet, threshold:int=5,contrast:float = 1.0,noise_level:float=0.2,noise_mag:float=0.025,opacity:float=1) -> str:
	img=import_image_base64(image_str)
	img=create_colored_image(img,grades=grades,kmean=kmean,palet=palet,threshold=threshold,contrast=contrast,noise_level=noise_level,noise_mag=noise_mag,opacity=opacity)
	return image_to_base64(img)



def k_means_for_colors(colors:np.ndarray,k:int=3) -> dict[int,int]:
	l=256/k/2
	centers=[(1+2*i)*l for i in range(k)]
	belonging:dict[int,int]={i:int((i)//(2*l)) for i,color in enumerate(colors)}

	old_belonging=belonging.copy()
	too_long=0
	while True:
		for affinity in set(belonging.values()):
			M=0
			c=0
			for i in belonging:
				if belonging[i]==affinity:
					M+=colors[i]
					c+=colors[i]*i
					
			
			if M==0:
				continue
			centers[affinity]=c/M

		for color in range(len(colors)):
			belonging[color]=centers.index(round_to_nearest(color,centers))

		if old_belonging==belonging:
			break
		old_belonging=belonging.copy()

		if too_long >100:
			break
		too_long+=1
	
	return belonging


def round_to_nearest(n:int,targets:list) -> int:
	return min(targets, key=lambda x:abs(x-n))



def paint_image_kmean(img_lum:np.ndarray,colors:dict[int,int],palet:list[list[int]],opacity:float=1) -> np.ndarray:
	img_layer=np.zeros((img_lum.shape[0],img_lum.shape[1],4),dtype=np.uint8)

	for i,row in enumerate(img_lum):
		for j,pix in enumerate(row):
			img_layer[i,j]=palet[colors[pix]]+[int(255*opacity)]

	return img_layer


def create_colored_image(image:np.ndarray,grades:int,kmean=False,palet=palet, threshold:int=5,contrast:float = 1.0,noise_level:float=0.2,noise_mag:float=0.025,opacity:float=1) -> np.ndarray:

	img=image
	img=image_preprocessing(img,threshold=threshold,contrast=contrast,noise_level=noise_level,noise_mag=noise_mag)


	img_lum=rgb_to_luminance(img,grades=grades,clip=False)

	colors=get_quantity_of_each_color(img_lum)

	if kmean:
		colors=k_means_for_colors(colors,k=grades)
		img_layer=paint_image_kmean(img_lum,colors,palet,opacity=opacity)


	else:
		colors=get_quantity_of_each_color(img_lum)
		colors_temp=colors.flatten()
		colors_temp.sort()

		n_max=colors_temp[-round(grades/4)]
		for i in range(colors.size):
			if colors[i]<n_max/4:
				colors[i]=0
		

		

		img_layer=paint_image(img_lum, colors,palet,opacity=opacity)
		img_layer=replace_rare_colors_by_closest(img_lum, colors, img_layer)

	return img_layer


def replace_rare_colors_by_closest(img_lum:np.ndarray, colors:np.ndarray, img_layer:np.ndarray) -> np.ndarray:
	# it is not closest, we search moving by square spiral, but algorithm is much easier

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


def paint_image(img_lum:np.ndarray, colors:np.ndarray,palet:list[list[int]],opacity:float=1) -> np.ndarray:
	img_layer=np.zeros((img_lum.shape[0],img_lum.shape[1],4),dtype=np.uint8)

	for i,row in enumerate(img_lum):
		for j,pix in enumerate(row):
			if colors[pix]!=0:
				img_layer[i,j]=palet[pix]+[int(255*opacity)]
			else:
				img_layer[i,j]=[255,255,255,int(255*opacity)]
	
	return img_layer
	



if __name__ == '__main__':
	main()




