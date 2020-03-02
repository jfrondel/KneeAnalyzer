import numpy as np
import os
import cv2
from .processing import fill_points

def make_kernel(_type='average',_size=3,_sigma=0.5):
    choices = ['average','gaussian','grad_x','grad_y','c_grad_x','c_grad_y']
    if _type not in choices:
        raise ValueError('Invalid filter type!! Select one from %s' % choices)
    if _size%2 is 0 or _size==0:
        raise ValueError('Invalid filter size!! Must be an odd integer larger than zero!!')
    if _type == choices[0]:
        K = np.ones((_size,_size))*1/(_size**2)
    elif _type == choices[1]:
        K = np.zeros((_size,_size))
        c = _size//2
        for k in range(_size**2):
            y = k//_size
            x = k%_size
            nom = (y-c)**2+(x-c)**2
            denom = -2*_sigma**2+1e-9
            K[y,x] = np.exp(nom/denom)
        K /= K.sum()
    if _type == choices[2]:
        K = np.array([-1,1]).reshape(1,-1)
    if _type == choices[3]:
        K = np.array([-1,1]).reshape(-1,1)
    if _type == choices[4]:
        K = np.array([-1,0,1]).reshape(1,-1)
    if _type == choices[5]:
        K = np.array([-1,0,1]).reshape(-1,1)
    return K

def make_shape_kernel(shapes):
    if type(shapes) == list:
        #Get minimum and maximum coordinates
        min_= np.array([1e9,1e9]); max_= np.array([-1e9,-1e9]);
        for shape in shapes:
            tmp_min = shape.min(0)
            if tmp_min[0] < min_[0]:
                min_[0] = tmp_min[0]
            if tmp_min[1] < min_[1]:
                min_[1] = tmp_min[1]
            tmp_max = shape.max(0)
            if tmp_max[0] > max_[0]:
                max_[0] = tmp_max[0]
            if tmp_max[1] > max_[1]:
                max_[1] = tmp_max[1]

        #Compute scale
        scale = max_ - min_
        #Create empty array for the filter (size is scale, padded with zeros)
        kernel = np.zeros(scale.astype(int)+20)

        #Iterate over the shapes and add them to the kernel
        for shape in shapes:
            #Compute new position for the shape
            shape_ = shape-min_+10
            shape_ = fill_points(shape_)

            #Iterate over the shape and set ones to the kernel
            for x,y in shape_:
                kernel[int(x),int(y)] = 1
                
    elif type(shapes) == np.ndarray:
        #Get minimum and maximum coordinates
        min_,max_ = shapes.min(0),shapes.max(0)
        #Compute scale
        scale = max_ - min_
        #Create empty array for the filter (size is scale, padded with zeros)
        kernel = np.zeros(scale.astype(int)+20)

        #Compute new position for the shape
        shape_ = shapes-min_+10
        shape_ = fill_points(shape_)


        #Iterate over the shape and set ones to the kernel
        for x,y in shape_:
            kernel[int(x),int(y)] = 1
            
    #Dilate the mask
    kernel = cv2.dilate(kernel,np.ones((3,3)))
    
        
    return kernel.T

def image_padding(image,pady,padx,c=0,mode='constant'):
    choices = ['constant','reflect']
    if mode not in choices:
        raise ValueError("Invalid mode argument!! Mode must be one from %s ." % choices)
    
    #Horizontal padding
    h,w = image.shape
    if mode == choices[0]:
        padding = np.zeros((h,padx))
        image = np.concatenate((padding,image,padding),1)
    elif mode == choices[1]:
        for _ in range(padx):
            try:
                paddingx0 = np.concatenate((paddingx0,image[:,0].reshape(-1,1)),1)
                paddingx1 = np.concatenate((paddingx1,image[:,-1].reshape(-1,1)),1)
            except NameError:
                paddingx0 = image[:,0].reshape(-1,1)
                paddingx1 = image[:,-1].reshape(-1,1)
        if padx > 0:
            image = np.concatenate((paddingx0,image,paddingx1),1)
    #Vertical padding
    h,w = image.shape
    if mode == choices[0]:
        padding = np.zeros((pady,w))
        image = np.concatenate((padding,image,padding),0)
    elif mode == choices[1]:
        for _ in range(pady):
            try:
                paddingy0 = np.concatenate((paddingy0,image[0,:].reshape(1,-1)),0)
                paddingy1 = np.concatenate((paddingy1,image[-1,:].reshape(1,-1)),0)
            except:
                paddingy0 = image[0,:].reshape(1,-1)
                paddingy1 = image[-1,:].reshape(1,-1)
        if pady > 0:
            image = np.concatenate((paddingy0,image,paddingy1),0)
    
    return image

def imfilter(image,kernel,mode='constant'):
    dims = image.shape
    #Zero pad the image
    kh,kw = kernel.shape
    I = image_padding(image,kh//2,kw//2,mode=mode)
    #Filter
    I = cv2.filter2D(I,-1,kernel)
    #Remove zeros
    I = I[kh//2:kh//2+dims[0],kw//2:kw//2+dims[1]]
    
    return I

def make_dog(image,size,sigma1,sigma2,mode='constant'):
    K1 = make_kernel('gaussian', size, sigma1)
    K2 = make_kernel('gaussian', size, sigma2)
    I1 = imfilter(image,K1,mode='constant')
    I2 = imfilter(image,K2,mode='constant')
    
    D = I1-I2
    return D

def make_grads(image,mode='constant'):
    Kx = make_kernel('c_grad_x')
    Ky = make_kernel('c_grad_y')
    Dx = imfilter(image, Kx, mode=mode)
    Dy = imfilter(image, Ky, mode=mode)
    
    return Dx,Dy

def make_grads_mag_theta(image,mode='constant'):    
    Dx,Dy = make_grads(image,mode=mode)
    mag = (Dx**2+Dy**2)**0.5
    theta = np.arctan(Dy/(Dx+1e-9))
    return mag,theta

def closing(image,kernel_size=3,n_iters=1):
    K = np.ones((kernel_size,kernel_size))
    for _ in range(n_iters):    
        image = cv2.dilate(image,K)
        image = cv2.erode(image,K)
    
    return image

def opening(image,kernel_size=3,n_iters=1):
    K = np.ones((kernel_size,kernel_size))
    for _ in range(n_iters):    
        image = cv2.erode(image,K)
        image = cv2.dilate(image,K)
    
    return image

def bw_area_open(image,area,connectivity=8):
    image = ((image>0)*255).astype(np.uint8)
    N,L,stats,centroids = cv2.connectedComponentsWithStats(image,connectivity=connectivity)
    inds = stats[:,-1]>area
    nonzeros = np.nonzero(inds)
    labels = np.linspace(0,N-1,N)
    labels = labels[nonzeros]
    mask = np.zeros(image.shape)
    for label in labels:
        mask += (L==label)*L
        
    mask = mask>0
    
    return mask

def canny(image,thres=0.5,small_thres=0.4,thres_type='maximum'):
    
    if image.max() > 1.0 or image.min()<0:
        raise TypeError("Input image has to be float, valued in range [0, 1]")
        
    thres_types = ['percentile','maximum']
    if thres_type not in thres_types:
        raise ValueError("Incorrect threshold type!! use one of the following: %s" % thres_types)
        
    if thres_type == thres_types[0]:
        tmax = np.percentile(image,thres*100)*255
    elif thres_type == thres_types[1]:
        tmax = thres*255
    tmin = 0.4*small_thres
    E = cv2.Canny((image*255).astype(np.uint8),int(tmin),int(tmax))
    
    return E

def get_max_location(image):
    max_val = image.max()
    ismax = image==max_val
    inds = np.nonzero(ismax)
    inds = np.array(inds).squeeze().T
    if len(inds.shape) == 1:
        idx = inds
    elif len(inds.shape)>1:
        dists = np.ones((inds.shape[0]))*1e9
        c = np.array(image.shape)*0.5
        for k in range(inds.shape[0]):
            d = (c-inds[k,:])**2
            dists[k] = d.sum()
        x = np.linspace(0,dists.shape[0]-1,dists.shape[0]).astype(int)
        idx = inds[np.argmin(dists)]
    return idx

def binarize(image):
    dog = make_dog(image,5,1,5)
    t = np.percentile(dog,80)
    bw = ((dog>t)*255).astype(np.uint8)
    bw = closing(bw)
    bw = bw_area_open(bw,20)
    E = canny(image,thres=0.5,small_thres=0.8)
    E = cv2.dilate(E,np.ones((3,3)))
    
    return bw*E

def gamma_preprocess(image,prctl=70,gamma=1.5,use_prctile=True):
    if use_prctile:
        t = np.percentile(image,prctl)
        bw = image>t
        gamma_corrected = (bw*image)**gamma
        I = image*(1-bw)+gamma_corrected
    else:
        I = image**gamma
    I = (I-I.min())/(I.max()-I.min())
    return I

def grad_1d(data,r=1):
    d1,d2 = [],[];
    for _ in range(r):
        d1.append(data[0])
        d2.append(data[-1])
    d1 = np.array(d1); d2 = np.array(d2);
    data = np.concatenate((d1,data,d2))
    
    output = np.zeros(data.shape[0]-2*r)
    
    n_steps = (data.shape[0]-2*r)//r
    
    for k in range(1,n_steps,1):
        output[k*r] = data[(k+1)*r]-data[(k-1)*r]
       
    return output[r:-r]

def graythresh(im):
    mu = im.mean()
    
    bw = (im>mu)*1.0
    
    while True:
        bw1 = im>=mu
        bw2 = im<mu
        mu1 = (im*bw1).sum()/bw1.sum()
        mu2 = (im*bw2).sum()/bw2.sum()
        
        mu_ = (mu1+mu2)/2
        
        if mu_==mu:
            break
        else:
            mu = mu_
            
        bw = (im>mu)*1.0
            
    return bw,mu

def create_mask(points,size_,n_contours=2,flips=[False,True]):
    N,_ = points.shape
    L = N//n_contours
    for k in range(n_contours):
        if k == 0:
            tmp = points[:L,:]
            if flips[k]:
                tmp = np.flip(tmp,0)
        elif k == n_contours-1:
            tmp_ = points[k*L:,:]
            if flips[k]:
                tmp_ = np.flip(tmp_,0)
            tmp = np.concatenate((tmp,tmp_),0)
        else:
            tmp_ = points[k*L:(k+1)*L,:]
            if flips[k]:
                tmp_ = np.flip(tmp_,0)
            tmp = np.concatenate((tmp,tmp_),0)
    points = tmp.astype(np.int32)
    mask = np.zeros(size_).astype(np.uint8)
    mask = cv2.fillPoly(mask,[points],1)
    
    return mask
