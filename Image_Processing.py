import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
import random
import numpy as np
import natsort
import math


class preprocess():
	
	 
 	   
    def __init__(self, dataset_location="", batch_size=1,shuffle=False):
    	self.location=dataset_location
    	self.batch_size=batch_size
    	list=os.listdir(self.location)
    	self.length=len(list)
    	self.seed = 2
    	self.shuffle =shuffle
    
    def rescale(self,s):
        import math
        dict_imgs=self.__getitem__()
        new_dict={}
        for im in dict_imgs:
            arr=mpimg.imread(dict_imgs[im])
            height=round(arr.shape[0]*s)
            width=round(arr.shape[1]*s)
            type=str(arr.dtype)
            img_height, img_width =arr.shape[:2]
            if len(arr.shape) > 2 and len(arr.shape)==3:
                shape1 = (height,width,3)
            elif len(arr.shape) == 2:
                shape1 = (height,width)
            resized = np.empty(shape1)
            resized = resized.astype(type)
            x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
            y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0
            for i in range(height):
                for j in range(width):
                    x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                    x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)
                    x_weight = (x_ratio * j) - x_l
                    y_weight = (y_ratio * i) - y_l
                    a = arr[y_l, x_l]
                    b = arr[y_l, x_h]
                    c = arr[y_h, x_l]
                    d = arr[y_h, x_h]
                    pixel = a * (1 - x_weight) * (1 - y_weight)+ b * x_weight * (1 - y_weight)+c * y_weight * (1 - x_weight)+d * x_weight * y_weight
                    resized[i,j] = pixel
            new_dict[im]=resized
        return new_dict
      
      
    def resize(self,h,w):
        img_dict=self.__getitem__()
        new_dict={}
        h_new=h
        w_new=w
        for k in img_dict:
            arr = mpimg.imread(img_dict[k])
            type=str(arr.dtype)
            height,width=arr.shape[:2]
            h_ratio=h_new/(height)
            w_ratio=w_new/(width)
            if len(arr.shape)==2:
                req=np.zeros((h_new,w_new),dtype=type)
            elif len(arr.shape)==3:
                req=np.zeros((h_new,w_new,3),dtype=type)
            
            for i in range(h_new):
                for j in range(w_new):
                    h_req = int(i / h_ratio)
                    w_req = int(j / w_ratio)
                    if len(arr.shape)==2:
                        req[i,j]=arr[h_req,w_req]
                    elif len(arr.shape)==3:
                        req[i,j,:]=arr[h_req,w_req,:]
                    
            new_dict[k]=req
        return new_dict  
         
	                        
    def translate(self,tx,ty):
    	dict_img=self.__getitem__()
    	new_dicttranslate={}
    	for i in dict_img:
    		arr=mpimg.imread(dict_img[i])
    		type=str(arr.dtype)
    		arr1=arr[::-1,:]
    		arr_translate=np.zeros(arr.shape,dtype=type)
    		arr_translate[ty:,tx:]=arr1[:arr1.shape[0]-ty,:arr1.shape[1]-tx]
    		new_dicttranslate[i]=arr_translate[::-1,:]
    	return new_dicttranslate

    		
    def crop(self,id1,id2,id3,id4):
    	dict=self.__getitem__()
    	
    	finaldict={}
    	for i in dict:
    		arr=mpimg.imread(dict[i])
    		type=str(arr.dtype)
    		id1n=(id1[0],arr.shape[0]-id1[1])
    		id2n=(id2[0],arr.shape[0]-id2[1])
    		id3n=(id3[0],arr.shape[0]-id3[1])
    		id4n=(id4[0],arr.shape[0]-id4[1])
    		finaldict[i]=arr[id1n[1]:id4n[1]+1,id1n[0]:id2n[0]+1]
    		
    	return finaldict

 
    def blur(self):
    	dict_img=self.__getitem__()
    	newdict={}
    	for i in dict_img:
    		arr=mpimg.imread(dict_img[i])
    		type=str(arr.dtype)
    		temp_arr=np.zeros(arr.shape,dtype=type)
    		for j in range(0,arr.shape[0]):
    			for k in range(0,arr.shape[1]):
    				#temp_arr.flags.writeable=True
    				if len(arr.shape)==2:
    					if j<arr.shape[0]-1 and k<arr.shape[1]-1 and j>0 and k>0:
    						temp_arr[j,k]=np.median(arr[j-1:j+2,k-1:k+2])
    					else:
    						temp_arr[j,k]=arr[j,k]
    				elif len(arr.shape)==3:
    					if j<arr.shape[0]-1 and k<arr.shape[1]-1 and j>0 and k>0:
    						temp_arr[j,k,0]=np.median(arr[j-1:j+2,k-1:k+2,0])
    						temp_arr[j,k,1]=np.median(arr[j-1:j+2,k-1:k+2,1])
    						temp_arr[j,k,2]=np.median(arr[j-1:j+2,k-1:k+2,2])
    					else:
    						temp_arr[j,k,:]=arr[j,k,:]
    		newdict[i]=temp_arr
    	return newdict
 
 
    def edge_detection(self):
    	dict=self.__getitem__()
    	finaldict={}
    	for k in dict:
    		arr=mpimg.imread(dict[k])
    		type=str(arr.dtype)
    		if len(arr.shape)==3:
    			arr=0.2989*(arr[:,:,0])+0.5870*(arr[:,:,1])+0.1140*(arr[:,:,2])	
    		Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    		Gy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    		narr=np.zeros((arr.shape[0]+2,arr.shape[1]+2))
    		narr[1:(-1),1:(-1)]=arr
    		Cx=np.zeros(arr.shape)
    		Cy=np.zeros(arr.shape)
    		for i in range(arr.shape[0]):
    			for j in range(arr.shape[1]):
    				Cx[i,j]=np.sum((Gx)*(narr[i:i+3,j:j+3]))
    				Cy[i,j]=np.sum((Gy)*(narr[i:i+3,j:j+3]))
    		C=((Cx**2)+(Cy**2))**0.5
    		C%=255
    		finaldict[k]=C
    	return finaldict
    	
    	
    def __getitem__(self):
    	list=natsort.natsorted(os.listdir(self.location))
    	if self.shuffle==True:
    		random.seed(self.seed)
    		random.shuffle(list)
    	dict={}
    	for i in range(self.batch_size):
    		val=str(self.location)+"/"+str(list[i])
    		key=list[i].split('.')[0]
    		dict[key]=val
    	return dict
    			
 
    def rgb2gray(self):
    	dict=self.__getitem__()
    	finaldict={}
    	for k in dict:
    		arr=mpimg.imread(dict[k])
    		t=arr.shape
    		if len(t)==3:
    			grey_arr=0.2989*(arr[:,:,0])+0.5870*(arr[:,:,1])+0.1140*(arr[:,:,2])
    		elif len(t)==2:
    			grey_arr=arr
    		finaldict[k]=grey_arr
    	return finaldict
    			
 
    def rotate(self,theta):
    	dict=self.__getitem__()
    	finaldict={}
    	for k in dict:
    		arr=mpimg.imread(dict[k])
    		type=str(arr.dtype)
    		cosine = np.cos(np.radians(theta))
    		sine = np.sin(np.radians(theta))
    		height = arr.shape[0]
    		width = arr.shape[1]
    		new_height = round(abs(arr.shape[0] * cosine) + abs(arr.shape[1] * sine)) + 1
    		new_width = round(abs(arr.shape[1] * cosine) + abs(arr.shape[0] * sine)) + 1
    		if len(arr.shape)==2:
    			output = np.zeros((int(new_height), int(new_width)),dtype=type)
    		elif len(arr.shape)==3:
    			output = np.zeros((int(new_height), int(new_width),3),dtype=type)
    			
    		original_centre_height = round(((arr.shape[0] + 1) / 2) - 1)
    		original_centre_width = round(((arr.shape[1] + 1) / 2) - 1)
    		new_centre_height = round(((new_height + 1) / 2) - 1)
    		new_centre_width = round(((new_width + 1) / 2) - 1)
    		for i in range(height):
    			for j in range(width):
    				y = arr.shape[0] - 1 - i - original_centre_height
    				x = arr.shape[1] - 1 - j - original_centre_width
    				new_y = math.floor(-x * sine + y * cosine)
    				new_x = math.floor(x * cosine + y * sine)
    				new_y = new_centre_height - new_y
    				new_x = new_centre_width - new_x
    				if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x >= 0 and new_y >= 0:
    					if len(arr.shape)==2:
    						output[int(new_y), int(new_x)] = arr[i, j]
    					elif len(arr.shape)==3:
    						output[int(new_y), int(new_x),:] = arr[i, j,:]
    						
    		finaldict[k]=output
    	return finaldict
    
    def contrast_stretching(self,min_val_r,max_val_r,min_val_s,max_val_s):
        import numpy as np
        from PIL import Image
        
        dicto=self.__getitem__()
        finaldict={}
        for k in dicto:
            
            input_image=dicto[k]
        
            img_array=Image.open(input_image)
            img_array=np.array(img_array)

            m1=(min_val_s/min_val_r)
            m2=(max_val_s-min_val_s)/(max_val_r-min_val_r)
            m3=(255-max_val_s)/(255-max_val_r)

            c2=max_val_s-(m2*max_val_r)
            c3=max_val_s-(m3*max_val_r)

            #To find the dimension of image
            n=len(img_array.shape)
            rgb=(n==3 and (img_array.shape)[-1]==3)
            if rgb:
                planes=3
            else:
                planes=1

            if planes==1:
                dimension=img_array.shape
                rows=dimension[0]
                cols=dimension[1]
                for i in range(rows):
                    for j in range(cols):
                        if img_array[i,j]<min_val_r and img_array[i,j]>=0:
                            img_array[i,j]=m1*img_array[i,j]
                        elif img_array[i,j]>=min_val_r and img_array[i,j]<max_val_r:
                            img_array[i,j]=(m2*img_array[i,j])+(c2)
                        elif img_array[i,j]>=max_val_r and img_array[i,j]<=255:
                            img_array[i,j]=(m3*img_array[i,j])+(c3)

            elif planes==3:
                dimension=img_array.shape
                rows=dimension[0]
                cols=dimension[1]
                for i in range(rows):
                    for j in range(cols):
                        for k in range(3):
                            if img_array[i,j,k]<min_val_r and img_array[i,j,k]>=0:
                                img_array[i,j,k]=m1*img_array[i,j,k]
                            elif img_array[i,j,k]>=min_val_r and img_array[i,j,k]<max_val_r:
                                img_array[i,j,k]=(m2*img_array[i,j,k])+(c2)
                            elif img_array[i,j,k]>=max_val_r and img_array[i,j,k]<=255:
                                img_array[i,j,k]=(m3*img_array[i,j,k])+(c3)

            output_image=img_array.astype('int64')
            finaldict[k]=output_image
        return finaldict


    def gamma_transform(self,gamma):
        from PIL import Image
        import numpy as np
        
        dicto=self.__getitem__()
        finaldict={}
        for k in dicto:
            
            input_image=dicto[k]
            img_array=Image.open(input_image)
            img_array=np.array(img_array)
            img_array=img_array/255
            output_image=((img_array)**gamma)*255
            output_image=output_image.astype('int64')
            output_image[output_image>255]=255
            finaldict[k]=output_image
        return finaldict
    
    def histogram_equalization(self):
    
        from PIL import Image
        import numpy as np
        
        dicto=self.__getitem__()
        finaldict={}
        for k in dicto:
            input_image=dicto[k]
            img_array=Image.open(input_image)
            img_array=np.asarray(img_array)

            #To find the dimension of image
            n=len(img_array.shape)
            rgb=(n==3 and (img_array.shape)[-1]==3)
            if rgb:
                planes=3
            else:
                planes=1
            rows=(img_array.shape)[0]
            cols=(img_array.shape)[1]
            eq_array=np.copy(img_array)

            #1D image
            if planes==1:
                hist=np.zeros((1,256))
                for i in range(256):
                    hist[0,i]=np.sum(img_array==i)
                hist_pdf=hist/(rows*cols)
                hist_cdf=np.copy(hist_pdf)
                for i in range(1,256):
                    hist_cdf[0,i]=hist_cdf[0,i-1]+hist_pdf[0,i]
                rel=255*hist_cdf
                rel=np.round_(rel)
                rel[rel>255]=255
                rel=rel.astype("uint8")
                for i in range(256):
                    eq_array[img_array==i]=(rel[0,i])
                finaldict[k]=eq_array

            #3D image
            elif planes==3:
                hist=np.zeros((3,256))
                for x in range(planes):
                    for i in range(256):
                        hist[x,i]=np.sum(img_array[:,:,x]==i)
                hist_pdf=hist/(rows*cols)
                hist_cdf=np.copy(hist_pdf)
                for x in range(planes):
                    for i in range(1,256):
                        hist_cdf[x,i]=hist_cdf[x,i-1]+hist_pdf[x,i]
                rel=255*hist_cdf
                rel=np.round_(rel)
                rel[rel>255]=255
                rel=rel.astype("uint8")
                for x in range(planes):
                    for i in range(rows):
                        for j in range(cols):
                            ip=img_array[i,j,x]
                            op=rel[x,ip]
                            eq_array[i,j,x]=op
                finaldict[k]=eq_array
        return finaldict
                
    
    def DWT_Haar(self):
        import numpy as np
        from PIL import Image
        
        dicto=self.__getitem__()
        finaldict={}
        for k in dicto:
            input_image=dicto[k]
            img=Image.open(input_image)
            img=np.asarray(img)

            #input and output shapes
            x,y=img.shape
            m,n=int(x/2),int(y/2)

            #Horizontal
            out1=np.zeros((x,y))
            lp=np.array([1,1])
            hp=np.array([1,-1])

            i,j=0,0
            k,l=0,0
            runs=0
            while i<x:
                while runs<=1:
                    while j<y-1:
                        if runs==0:
                            out1[i,l]=np.sum(lp*img[i,j:j+2])
                        else:
                            out1[i,l]=np.sum(hp*img[i,j:j+2])
                        j+=2
                        l+=1
                    j=0
                    runs+=1
                l=0
                runs=0
                i+=1

            #Vertical
            out2=np.zeros((x,y))
            lp=np.transpose(lp)
            hp=np.transpose(hp)

            i,j=0,0
            k,l=0,0
            runs=0
            while j<y:
                while runs<=1:
                    while i<x-1:
                        if runs==0:
                            out2[k,j]=np.sum(lp*out1[i:i+2,j])
                        else:
                            out2[k,j]=np.sum(hp*out1[i:i+2,j])
                        i+=2
                        k+=1
                    i=0
                    runs+=1
                k=0
                runs=0
                j+=1
            finaldict[k]=out2
        return finaldict
    
    def save_output(self,req_location,dictionary):
        for im in dictionary:
            arr=dictionary[im]
            img=Image.fromarray(arr)
            new_location=req_location+"/"+str(im)+".png"
            img.save(new_location)
