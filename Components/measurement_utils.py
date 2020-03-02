import numpy as np
import os
import cv2
import pandas as pd
from copy import deepcopy

from .asm_utils import ASM,refine_joint_location,find_normal,find_grad_max
from .image_processing import *
from .IO_utils import csv_to_asm,csv_to_clustered_arrays,asm_to_csv,points_to_csv
from .processing import stratified_split

def get_list_inds(content,values):
    idx = []
    for k in range(len(content)):
        for val in values:
            if content[k] == val:
                idx.append(k)
    return idx

def crop_shape(shape,start=2,stop=7,from_=[1,-1]):
    for d in from_:
        if d == 1:
            tmp = shape[start:stop,:]
        elif d == -1:
            N = shape.shape[0]
            tmp = shape[N-stop:N-start,:]
        try:
            out = np.concatenate((out,tmp),0)
        except NameError:
            out = tmp
    return out

def bw_classifier(bw,contours):
    #Get nonzero indices from binary image
    inds = np.array(np.nonzero(bw))
    inds = deepcopy(np.flip(inds.T,1))    
    
    outLT,outMT,outLF,outMF = [],[],[],[]
    
    for idx in inds:
        #Get distance to contour points
        Dlt = (contours[0]-idx)**2
        Dlt = (Dlt.sum(1))**0.5
        
        Dmt = (contours[1]-idx)**2
        Dmt = (Dmt.sum(1))**0.5
        
        Dlf = (contours[2]-idx)**2
        Dlf = (Dlf.sum(1))**0.5
        
        Dmf = (contours[3]-idx)**2
        Dmf = (Dmf.sum(1))**0.5
        
        mins = [Dlt.min(),Dmt.min(),Dlf.min(),Dmf.min()]
        
        argmin = np.argmin(mins)
        
        if argmin == 0:
            outLT.append(idx)
        elif argmin == 1:
            outMT.append(idx)
        elif argmin == 2:
            outLF.append(idx)
        elif argmin == 3:
            outMF.append(idx)
            
    return np.array(outLT),np.array(outMT),np.array(outLF),np.array(outMF)

def get_contour_neighbours(points,contour):
    out = []
    points = fill_points(points)
    for p in points:
        D = (contour-p)**2
        D = D.sum(1)**0.5
        idx = np.argmin(D)
        out.append(contour[idx,:])
        
    out = fill_points(np.array(out))
    return out

def js_contour_maker(bw,joint):
    #Get binary image shape
    h,w = bw.shape    
    
    #Classify edges
    lateral_tibia_,medial_tibia_,lateral_femur_,medial_femur_ = bw_classifier(bw,joint)
    
    #Clean the contours
    lateral_tibia = get_contour_neighbours(joint[0],lateral_tibia_)
    medial_tibia = get_contour_neighbours(joint[1],medial_tibia_)
    lateral_femur = get_contour_neighbours(joint[2],lateral_femur_)
    medial_femur = get_contour_neighbours(joint[3],medial_femur_)
    
    return lateral_tibia,medial_tibia,lateral_femur,medial_femur

def crop_points_xaxis(points,xstart,xstop):
    out = []
    for p in points:
        if p[0]>=xstart and p[0]<= xstop:
            out.append(p)
    out = np.array(out)
    return out

def joint_edge_neighbours(contour1,contour2):
    D = 1e9; points = np.zeros((2,2))
    for p in contour1:
        d = (contour2-p)**2
        d = d.sum(1)**0.5
        min_idx = np.argmin(d)
        if d[min_idx] < D:
            D = d[min_idx]
            points[0,:] = p
            points[1,:] = contour2[min_idx,:]
            
    return D,points

def disk_measure(image,Rmax=50):
    h,w = image.shape
    out = np.zeros((h,w,Rmax))
    for k in range(Rmax,0,-1):
        #Make disk kernel
        kernel = make_kernel(_type='gaussian',_size=2*k+1,_sigma=k)
        kernel = (kernel-kernel.min())/(kernel.max()-kernel.min())
        kernel = (kernel>=np.exp(-1))*1.0
        #Convolve with input image
        im = imfilter(image,kernel)
        #Get regions where the disk doesn't overlap with the foreground
        im = (im<1)*k
        out[:,:,k-1] = im
        
    #Get maximum radii
    out = np.array(out)
    out = np.argmax(out,axis=2)+1
    
    return out

def joint_measure(image,joint, landmarks=[3,8]):
    h,w = image.shape
    
    #Crop area surrounding the joint
    min_ = joint.min(0); max_ = joint.max(0);
    x0 = min_[0];x1 = max_[0];y0 = min_[1];y1 = max_[1];
    y0 = np.max((0,y0-50));y1 = np.min((h,y1+50));
    x0 = int(x0);x1 = int(x1);y0 = int(y0);y1 = int(y1);
    crop = image[y0:y1,x0:x1]
    
    h_,w_ = crop.shape
    #Split the cropped image to 3 regions and binarize
    bw0,_ = graythresh(crop[:,:w_//3])
    bw1,_ = graythresh(crop[:,w_//3:2*w_//3])
    bw2,_ = graythresh(crop[:,2*w_//3:])
    
    #Combine to single image
    bw = np.concatenate((bw0,bw1,bw2),1)
    
    #Remove small components from the binary image
    bw = bw_area_open(bw,20)
    
    #Fit disks and measure the joint
    radii = disk_measure(bw)
    
    #Select landmarks around lateral and medial tibial plateu
    joint_ = joint-np.array([x0,y0])
    tibia = joint_[:joint_.shape[0]//2,:];femur = joint_[joint_.shape[0]//2:,:];
    nt = tibia.shape[0];nf = femur.shape[0];
    
    lateral = np.concatenate((tibia[landmarks[0]:landmarks[1],:],femur[landmarks[0]:landmarks[1],:]),0)
    medial = np.concatenate((tibia[nt-landmarks[1]:nt-landmarks[0],:],femur[nf-landmarks[1]:nf-landmarks[0],:]),0)

    #Remove the measurements outside of the joint contour
    maskLat = create_mask(lateral,crop.shape)
    maskMed = create_mask(medial,crop.shape)
    maskFull = create_mask(joint_,crop.shape)
    radiiLat = radii*maskLat
    radiiMed = radii*maskMed
    radiiFull = radii*maskFull
    
    output = np.concatenate((radiiFull.reshape(h_,w_,1),
                            radiiLat.reshape(h_,w_,1),
                            radiiMed.reshape(h_,w_,1)),2)
    masks = np.concatenate((maskFull.reshape(h_,w_,1),
                            maskLat.reshape(h_,w_,1),
                            maskMed.reshape(h_,w_,1)),2)
    
    coords = np.array([x0,y0])
    
    return output,masks,coords

def get_width(radii,mask,resolution=0.148):
    vals = [];
    h,w = radii.shape
    #Scan radius mask, and get maxima along vertical axis
    vals = []
    for k in range(w):
        val,idx = radii[:,k].max(),np.argmax(radii[:,k])
        if mask[idx,k] != 0:
            vals.append(val)
    '''
    for k in range(h*w):
        y = k//w; x=k%w;
        if mask[y,x] != 0:
            vals.append(radii_[y,x])
    '''
    vals = (np.array(vals)*2+1).astype(np.float)
    vals *= resolution
    return vals.min(),vals.max(),vals.mean(),np.median(vals)

def get_joint_edges(image,joint,landmarks=[2,7]):
    h,w = image.shape
    
    #Crop area surrounding the joint
    min_ = joint.min(0); max_ = joint.max(0);
    x0 = min_[0];x1 = max_[0];y0 = min_[1];y1 = max_[1];
    y0 = np.max((0,y0-50));y1 = np.min((h,y1+50));
    x0 = int(x0);x1 = int(x1);y0 = int(y0);y1 = int(y1);
    crop = image[y0:y1,x0:x1]
    
    h_,w_ = crop.shape
    #Split the cropped image to 3 regions and binarize
    bw0,_ = graythresh(crop[:,:w_//3])
    bw1,_ = graythresh(crop[:,w_//3:2*w_//3])
    bw2,_ = graythresh(crop[:,2*w_//3:])
    
    #Combine to single image
    bw = np.concatenate((bw0,bw1,bw2),1)
    
    #Get edges
    mag,theta = make_grads_mag_theta(crop)
    
    t = np.percentile(mag,80)
    magbw = (mag>t)
    magbw = bw_area_open(magbw,16,8)
    
    #Get joint contours from edges
    N = joint.shape[0]
    N_ = N//4
    lattib = joint[:N_,:]-np.array([x0,y0]);
    medtib = joint[N_:2*N_,:]-np.array([x0,y0]);
    latfem = joint[2*N_:3*N_,:]-np.array([x0,y0]);
    medfem = joint[3*N_:,:]-np.array([x0,y0]);
    
    contours = [lattib,medtib,latfem,medfem]
    latTib,medTib,latFem,medFem = js_contour_maker(magbw,contours)
    
    
    #Select contours based on the selected landmarks
    x0lat = np.min((lattib[landmarks[0],0],latfem[landmarks[0],0]));
    x1lat = np.max((lattib[landmarks[1],0],latfem[landmarks[1],0]));
    x0med = np.min((medtib[(N_-1)-landmarks[1],0],medfem[(N_-1)-landmarks[1],0]));
    x1med = np.max((medtib[(N_-1)-landmarks[0],0],medfem[(N_-1)-landmarks[0],0]));
    
    latTib_cropped = crop_points_xaxis(latTib,x0lat,x1lat);
    medTib_cropped = crop_points_xaxis(medTib,x0med,x1med);
    latFem_cropped = crop_points_xaxis(latFem,x0lat,x1lat);
    medFem_cropped = crop_points_xaxis(medFem,x0med,x1med)
    
    return latTib_cropped,medTib_cropped,latFem_cropped,medFem_cropped,x0,y0

def joint_measure_from_edges(image,joint,landmarks=[2,7]):
    latTib,medTib,latFem,medFem,x,y = get_joint_edges(image,joint,landmarks)
    #Get nearest neighbours
    Dlat,pointslat = joint_edge_neighbours(latTib,latFem)
    Dmed,pointsmed = joint_edge_neighbours(medTib,medFem)
    
    P = np.array([x,y])
    return latTib+P,medTib+P,latFem+P,medFem+P,Dlat,Dmed,pointslat+P,pointsmed+P

class kneeModels(object):
    def __init__(self,datapath='./annotations',modelpath='./models',train=False,split=None,grade_file=None,train_size=0.50):
        #Path to model components
        self.modelpath = modelpath
        #Active shape models
        self.fullASM = None; self.tibiaASM = None;
        self.femurASM = None; self.jointASM = None;
        #Image processing components
        self.filter = None
        self.Dx = None; self.Dy = None;
        self.mag = None; self.theta = None;
        
        self.output = []
        
        if train:
            self.train_models(datapath,split,grade_file,train_size)
        else:
            path = os.path.join(self.modelpath,'ASM')
            files = os.listdir(path)
            self.load_models(path,files)
        return
    
    def __call__(self,image,shapes=['full','tibia','femur','joint'],R=[10,10,10,10],orientations=[[None],[None],[None],[None]],
                 n_iters=100,crop_size=120,resolution=0.148):
        #Clear output
        self.output = []
        #Generate binarized image, x/y gradient images, gradient magnitude image and gradient direcion image
        bw = binarize(image)
        self.Dx,self.Dy = make_grads(gamma_preprocess(image))
        self.mag,self.theta = make_grads_mag_theta(gamma_preprocess(image))
        #Locate the center of the joint
        idx = self.locate_joint(bw,self.filter)
        idx = np.flip(idx)
        
        
        new_idx,scale = refine_joint_location(image,idx)        
        
        #self.fullASM.update_scale(scale);self.tibiaASM.update_scale(scale);
        #self.femurASM.update_scale(scale);self.jointASM.update_scale(scale);
                
        #Apply full joint ASM
        #points,start = self.update_position('center',new_idx,None,self.fullASM,image,n_contours=2,
        #                     connect=False,R=R,n_iters=n_iters,orientations=['+d','+d','+d','-d','+d','-d','-d','-d'])
        
        for shape,R_,orientation in zip(shapes,R,orientations):
            if shape == 'full':
                points_,start = self.update_position('center',new_idx,None,self.fullASM,image,n_contours=2,
                                                    connect=False,R=R_,n_iters=n_iters,orientations=orientation)
                self.output.append(start);self.output.append(points_);
            elif shape == 'tibia':
                points = self.output[1]
                N = points.shape[0]
                points = points[:N//2,:]
                points_,_ = self.update_position('points',None,points,self.tibiaASM,image,n_contours=1,
                         connect=False,R=R_,n_iters=n_iters,orientations=orientation)
                self.output.append(points_)
            elif shape == 'femur':
                points = self.output[1]
                N = points.shape[0]
                points = points[N//2:,:]
                points_,_ = self.update_position('points',None,points,self.femurASM,image,n_contours=1,
                                 connect=False,R=R_,n_iters=n_iters,orientations=orientation)
                self.output.append(points_)
            elif shape == 'joint':
                #Apply joint ASM
                tib = self.output[2]; fem = self.output[3];
                Nt = tib.shape[0];Nf = fem.shape[0]
                points = np.concatenate((tib[Nt//4+1:3*Nt//4+1,:],fem[Nf//4+1:3*Nf//4+1,:]),0);
                points_,_ = self.update_position('points',None,points,self.jointASM,image,n_contours=2,
                                     connect=True,R=R_,n_iters=n_iters,orientations=orientation)
                self.output.append(points_)
                
        #Create joint space contour based on selected shapes
        if shapes[-1] != 'joint':
            if shapes[-1] == 'full':
                #Create joint if no other shapes will be delineated
                points_ = self.output[1]
            elif shapes[-1] == 'femur' and shapes[-2] == 'tibia':
                points_ = np.concatenate((self.output[-2],self.output[-1]),0)
            elif shapes[-2] == 'femur' and shapes[-1]=='tibia': 
                points_ = np.concatenate((self.output[-1],self.output[-2]),0)
            N = points_.shape[0]
            tibia = points_[:N//2,:];femur = points_[N//2:,:];
            nt = tibia.shape[0];nf=femur.shape[0];
            joint = np.concatenate((tibia[nt//4:3*nt//4,:],femur[nf//4:3*nf//4,:]),0)
            self.output.append(joint)            
        
        return self.output
    
    def locate_joint(self,image,kernel):
        joint_center_image = imfilter(image,kernel)
        
        idx = get_max_location(joint_center_image)
        return idx
    
    def update_position(self,mode,center,points,asm,image,n_contours=1,connect=False,R=10,n_iters=100,orientations=[None]):
        if mode == 'center':
            #Get shape from asm
            orig = asm.mean_shape.squeeze() 
            orig = orig.reshape(orig.shape[0]//2,2)

            tmp = asm.mean_shape.squeeze()
            tmp = tmp.reshape(tmp.shape[0]//2,2)

            #Move shape on the image
            tmp -= tmp.mean(0)
            tmp += center
        elif mode == 'points':
            orig = points
            tmp = points
        
        for k in range(n_iters):
            #Compute contour normals
            normal = find_normal(tmp,n_contours,connect)
            #Find gradient maxima along the contour normal
            new_points,grads = find_grad_max(self.Dx,self.Dy,self.mag,tmp,normal,R,orientations)
            
            #Move the ASM to the new position
            tmp_ = asm(new_points)
            
            #Compute change in the distance
            D = (tmp-tmp_)**2
            D = D.sum(1)**0.5
            D = D.mean()
            tmp = tmp_
            if D < 1:
                break
        
        return tmp,orig
    
    def load_contours(self,path,split):
        full,tibia,femur,joint = [],[],[],[]
        for file in split:
            fullfile = os.path.join(path,file+'.csv')
            tib,fem = csv_to_clustered_arrays(fullfile,)
            full.append(np.concatenate((tib,fem),0));tibia.append(tib);
            femur.append(fem);joint.append(np.concatenate((tib[10:30],fem[10:30]),0))
        return full,tibia,femur,joint
    
    def train_models(self,path,split,grade_file,train_size=0.50):
        #Set rng seed for reproducibility
        np.random.seed(42)
        if split == None and grade_file == None:
            raise ValueError("Specify files (split) for training the model or give path to a excel file containing the grades for automatic splitting.")
        if split == None and type(grade_file) == str:
            #Read to dataframe
            df = pd.read_excel(grade_file)
            names = np.array(df['Name'])
            grades = np.array(df['KL Grade'])
            
            #Remove ungraded samples
            nan_inds = np.isnan(grades)
            names,grades= names[~nan_inds],grades[~nan_inds]
            
            #Generate train/test split based on the grades
            split,grade_split,_ = stratified_split(names,grades,split_size=train_size)
                
        #Load points
        full,tibia,femur,joint = self.load_contours(path,split)
        
        #Train the models
        self.fullASM = ASM(full);
        self.tibiaASM = ASM(tibia);
        self.femurASM = ASM(femur);
        self.jointASM = ASM(joint);
        
        #Generate filter for the joint localization from the mean tibia and femur shapes
        _tibia = self.tibiaASM.mean_shape.squeeze(); _femur = self.femurASM.mean_shape.squeeze();
        _tibia = _tibia.reshape(_tibia.shape[0]//2,2);_femur = _femur.reshape(_femur.shape[0]//2,2)
        self.filter = make_shape_kernel([_tibia,_femur])
        
        #Save training data
        self.save_models(self.modelpath,['full','tibia','femur','joint'],
                         split,grade_split,[full,tibia,femur,joint],
                        [self.fullASM,self.tibiaASM,self.femurASM,self.jointASM])
            
        return
    
    def load_models(self,path,files,weights=[True,True,True,False]):
        for file,weighting in zip(files,weights):
            _asm = ASM(points=None,train=False,use_weights=weighting)
            _asm.load(os.path.join(path,file))
            if file.endswith('full.csv'):
                self.fullASM = _asm
            elif file.endswith('tibia.csv'):
                self.tibiaASM = _asm
            elif file.endswith('femur.csv'):
                self.femurASM = _asm
            elif file.endswith('joint.csv'):
                self.jointASM = _asm
                
        #Generate filter for the joint localization from the mean tibia and femur shapes
        _tmp = self.fullASM.mean_shape.squeeze();
        _tibia = _tmp[:_tmp.shape[0]//2]; _femur = _tmp[_tmp.shape[0]//2:]
        _tibia = _tibia.reshape(_tibia.shape[0]//2,2);_femur = _femur.reshape(_femur.shape[0]//2,2)
        self.filter = make_shape_kernel([_tibia,_femur])
        
        
        return
    
    def save_models(self,modelpath,names,datasplit,grades,points,active_shape_models):
        #Check if save folder exists
        if not os.path.isdir(modelpath):
            os.mkdir(modelpath)
        #Make folder for landmark and ASM data
        pointdir = os.path.join(modelpath,'Points')
        asmdir = os.path.join(modelpath,'ASM')
        if not os.path.isdir(pointdir):
            os.mkdir(pointdir)
        if not os.path.isdir(asmdir):
            os.mkdir(asmdir)
            
        #Save landmarks and ASM components
        for name,contours,asm in zip(names,points,active_shape_models):
            #Save directory
            _dir = os.path.join(pointdir,name)
            if not os.path.isdir(_dir):
                os.mkdir(_dir)
            for sample,contour in zip(datasplit,contours):
                filename = os.path.join(_dir,sample+'_'+name+'.csv')
                points_to_csv(filename,contour)
            _dir = asmdir
            if not os.path.isdir(_dir):
                os.mkdir(_dir)
            filename = os.path.join(_dir,'ASM_'+name+'.csv')
            asm.save(filename)
        #Save training split
        _dict = {'Name': datasplit,'KL Grade':grades}
        df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in _dict.items()})        
        #Write csv
        df.to_csv(os.path.join(self.modelpath,'train_split.csv'), sep='\t', encoding='utf-8', index = False)
            
        return
    
    def save_results(self,sample,names=['full','tibia','femur','joint']):
        #Create directory for results
        savepath = './results'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        #Create segmentation directory
        savepath = os.path.join(savepath,'segmentation')
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        #Make dictionary for the data
        _dict = {}
        for result,name in zip(self.results,names):
            _tmp = result
            _tmp = result.reshape(2*result.shape[0]).squeeze()
            _tmpdict = {name:_tmp}
            _dict.update(_tmpdict)
        #Make sure all arrays have the same length, use NaN fill
        df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in _dict.items()}) 
        #Write csv
        savename = os.path.join(savepath,sample+'.csv')
        df.to_csv(savename, sep='\t', encoding='utf-8', index = False)


def set_rois(points,w=95,h=41,dist=1,resolution = 0.148):
    "sb = 95*41, tr = 95*95"
    if len(points.shape)<2:
        points = points.reshape(1,-1)
    y = 0; x = 0;
    for p in points:
        if p[1] > y:
            y = p[1]; x = p[0];
    y += dist/resolution
    
    x0 = x-w//2; x1 = x0+w; y0 = y; y1 = y0+h;
    
    bbox1 = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    bbox2 = np.array([[x0,y1],[x1,y1],[x1,y1+95],[x0,y1+95]])
    
    return bbox1,bbox2

class kneeAnalyzer(object):
    def __init__(self,annotations='./annotations',models='./models',results='./results',js_from='joint',
                 shapes=['initial','full','tibia','femur','joint'],
                 orientations = [[None],['+dx','+dy','+dy','-dx'],['+dx','-dy','-dy','-dx'],['+dy','-dy']],
                 R=[60,10,10,20],n_iters=200,resolution=0.148,crop=120):
        self.annotations = annotations
        self.models = models
        self.results= results
        self.shapes = shapes
        self.js_from = js_from
        self.R = R
        self.ori = orientations
        self.n_iters = n_iters
        self.resolution = resolution
        self.crop = crop
        self.model = None
        return
        
    def train_models(self,grades,train_size=0.50):
        self.model = kneeModels(datapath=self.annotations,modelpath=self.models,train=True,grade_file=grades)
        return
    
    def load_models(self):
        self.model = kneeModels(datapath=self.annotations,modelpath=self.models,train=False)
        return
    
    def __call__(self,image,name):
        self.segmentation,self.measurements,self.lateral,self.medial,self.rois,self.edgeMeasurements = self.segment(image,name)
        
        self.save_segmentation(name,self.segmentation,self.shapes)
        self.save_JSW(name,self.lateral,self.medial)
        self.save_rois(name,self.rois)
        self.save_measure_areas(name, self.measurements[0], self.measurements[-1][0], self.measurements[-1][1])
        self.save_edge_measurements(name,self.edgeMeasurements[0], self.edgeMeasurements[1],
                                    self.edgeMeasurements[2], self.edgeMeasurements[3],
                                    self.edgeMeasurements[4], self.edgeMeasurements[5],
                                    self.edgeMeasurements[6],self.edgeMeasurements[7])
    
    def segment(self,image,name):
        #Normalize image
        image = (image-image.min())/(image.max()-image.min())
        #Segment images
        segmentation = self.model(image,shapes=self.shapes[1:],R=self.R,orientations=self.ori,
                                  n_iters=self.n_iters,crop_size=self.crop,resolution=self.resolution)
        
        #Measure JSW from ASM
        measurements = joint_measure(image,segmentation[-1])
        lateral = get_width(measurements[0][:,:,1],measurements[1][:,:,1],self.resolution)
        medial = get_width(measurements[0][:,:,2],measurements[1][:,:,2],self.resolution)
        
        #Measure JSW from gradient contours
        edgeMeasurements = joint_measure_from_edges(image,segmentation[-1],landmarks=[0,5])
        
        #Place ROI
        points = segmentation[-1]; N = points.shape[0]
        sbLeft,tLeft = set_rois(points[5,:],w=95,h=41);
        sbRight,tRight = set_rois(points[N//2-5,:],w=95,h=41)
        rois = [sbRight,sbLeft,tRight,tLeft]
        return segmentation,measurements,lateral,medial,rois, edgeMeasurements
    
    def measure_all(self,DICOM,exclusions=[],guiclass_=None):
        N = DICOM.get_number_of_files()
        for k in range(2*N):
            if guiclass_ != None:
                title = 'Analysis'; content = 'Analyzing knee {0} | {1}'.format(k+1,2*N)
                guiclass_.txt.set_content(title,content)
                guiclass_.xray.render_text(guiclass_.txt())
            if  k%2==0:
                side = 'R'
            else:
                side = 'L'
            name = DICOM.get_name(k//2)+'_'+side
            if name in exclusions:
                continue
            else:
                DICOM.read_file(k//2)
                image = DICOM.pixels(side)
                segmentation,measurements,lateral,medial,rois,edgeMeasurements = self.segment(image,name)
                self.save_segmentation(name,segmentation,self.shapes)
                self.save_JSW(name,lateral,medial)
                self.save_rois(name,rois)
                self.save_measure_areas(name, measurements[0], measurements[-1][0], measurements[-1][1])
                self.save_edge_measurements(name,edgeMeasurements[0],edgeMeasurements[1],edgeMeasurements[2],
                                            edgeMeasurements[3],edgeMeasurements[4],edgeMeasurements[5],
                                            edgeMeasurements[6],edgeMeasurements[7])
            
    def save_segmentation(self,name,segmentation,shapes,string=None):
        #Make directorues for save data
        if not os.path.isdir(self.results):
            os.mkdir(self.results)
        segment_dir = os.path.join(self.results,'segmentation')
        if not os.path.isdir(segment_dir):
            os.mkdir(segment_dir)
            
        #Save results
        for seg,shape in zip(segmentation,shapes):
            fullpath = os.path.join(segment_dir,shape)
            if not os.path.isdir(fullpath):
                os.mkdir(fullpath)
            if string != None:
                savename = name+'_'+shape+'_'+string+'.csv'
            else:
                savename = name+'_'+shape+'.csv'
            savepath = os.path.join(fullpath,savename)
            points_to_csv(savepath,seg)
            
        return
            
    def save_JSW(self,name,lateral,medial,string=None):
        #Make directorues for save data
        if not os.path.isdir(self.results):
            os.mkdir(self.results)
        width_dir = os.path.join(self.results,'JS_widths')
        if not os.path.isdir(width_dir):
            os.mkdir(width_dir)
        if string != None:
            savename = name+'_'+string+'.csv'
        else:
            savename = name+'.csv'
        savepath = os.path.join(width_dir,savename)
        #Generate dataset
        dict_ = {'Width':['Min','Max','Mean','Median'],'Lateral':lateral,'Medial':medial}
        df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in dict_.items()})        
        #Write csv
        df.to_csv(savepath, sep='\t', encoding='utf-8', index = False)
            
        return
    
    def save_measure_areas(self,name,mask,x,y,string=None):
        #Make directorues for save data
        if not os.path.isdir(self.results):
            os.mkdir(self.results)
        mask_dir = os.path.join(self.results,'JSW_masks')
        if not os.path.isdir(mask_dir):
            os.mkdir(mask_dir)
        if string != None:
            imagesavename = name+'_'+string+'.png'
            locationsavename = name+'_'+string+'.csv'
        else:
            imagesavename = name+'.png'
            locationsavename = name+'.csv'
        savepath = os.path.join(mask_dir,imagesavename)
        mask = mask.astype(np.uint8)
        cv2.imwrite(savepath,mask)
        savepath = os.path.join(mask_dir,locationsavename)
        dict_ = {'x':x,'y':y}
        df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in dict_.items()})        
        #Write csv
        df.to_csv(savepath, sep='\t', encoding='utf-8', index = False)
        
    def save_edge_measurements(self,name,latTibia,medTibia,latFemur,medFemur,
                                   DLateral,DMedial,p_lateral,p_medial,string=None):
        #Make directorues for save data
        if not os.path.isdir(self.results):
            os.mkdir(self.results)
        edge_dir = os.path.join(self.results,'JS_edges')
        if not os.path.isdir(edge_dir):
            os.mkdir(edge_dir)
        if string != None:
            savename = name+'_'+'edges'+'_'+string+'.csv'
        else:
            savename = name+'_'+'edges'+'.csv'
        savepath = os.path.join(edge_dir,savename)
        dict_ = {'tibia_lateral':latTibia.ravel(),'tibia_medial':medTibia.ravel(),
                 'femur_lateral':latFemur.ravel(),'femur_medial':medFemur.ravel()}
        
        df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in dict_.items()})
        df.to_csv(savepath,sep='\t',encoding='utf-8',index=False)
        
        if string != None:
            savename = name+'_'+'points'+'_'+string+'.csv'
        else:
            savename = name+'_'+'points'+'.csv'
        savepath = os.path.join(edge_dir,savename)
        dict_ = {'x_lateral':p_lateral[:,0],'y_lateral':p_lateral[:,1],
                 'x_medial':p_medial[:,0],'y_medial':p_medial[:,1]}
        df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in dict_.items()})
        df.to_csv(savepath,sep='\t',encoding='utf-8',index=False)
    
    def save_rois(self,name,rois,string=None):
        #Make directorues for save data
        if not os.path.isdir(self.results):
            os.mkdir(self.results)
        roi_dir = os.path.join(self.results,'ROIs')
        if not os.path.isdir(roi_dir):
            os.mkdir(roi_dir)
        if string != None:
            savename = name+'_'+string+'.csv'
        else:
            savename = name+'.csv'
        savepath = os.path.join(roi_dir,savename)
        #Generate dataset
        dict_ = {'bounding_box':['x0','y0','x1', 'y0','x1', 'y1','x0', 'y1'],'sbRight':rois[0].flatten(),'sbLeft':rois[1].flatten(),
                'tRight':rois[2].flatten(),'tLeft':rois[3].flatten()}
        df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in dict_.items()})        
        #Write csv
        df.to_csv(savepath, sep='\t', encoding='utf-8', index = False)
            
        return