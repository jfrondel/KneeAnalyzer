import numpy as np
import cv2
import pandas as pd
from copy import deepcopy
import os
import pydicom

from sklearn.cluster import KMeans

from .processing import get_points,get_points_as_lists, make_actor, make_point_actor, fill_points, sort_points
from Components.processing import get_line_ends, get_point_length

def get_forced_resolution(dicom,spatial = 0.148):
    pixels = dicom.pixel_array
    cur_spatial = np.float(dicom.SpatialResolution)
    if cur_spatial != spatial:
        scale = cur_spatial/spatial
        pixels = cv2.resize(pixels,(0,0),fx=scale,fy=scale)
        
    return pixels

class DICOM(object):
    def __init__(self):
        self.files = None
        self.path = None
        self.ds = None
        
    def get_name(self,idx=None):
        if idx is None:
            if type(self.files) is str:
                S = self.files.split('/')
                return S[-1]
            else:
                print('No file index!!')
        else:
            if type(self.files) is list:
                S = self.files[idx].split('/')
                return S[-1]
            else:
                print("Files are not in a list, don't use indexing!!")
                
    def get_number_of_files(self):
        if type(self.files) is str:
            return 1
        elif type(self.files) is list:
            return len(self.files)
        
    def read_file(self,idx=None):
        if idx is None:
            if type(self.files) is str:
                #fullfile = os.path.join(self.path,self.files)
                self.ds = pydicom.read_file(self.files)
            else:
                print("No file index!!")
        else:
            if type(self.files) is list:
                #fullfile = os.path.join(self.path,self.files[idx])
                self.ds = pydicom.read_file(self.files[idx])
            else:
                print("Files are not in a list, don't use indexing!!")        
        
    def pixels(self,side='R'):
        n_bits = self.ds.BitsAllocated
        monochrome = self.ds.PhotometricInterpretation
        pixels = get_forced_resolution(self.ds,0.148)
        size = pixels.shape
        if side == 'R':
            im = pixels[:,:size[1]//2]
        elif side == 'L':
            im = pixels[:,size[1]//2:]
            im = deepcopy(np.flip(im,1))
        if monochrome == 'MONOCHROME2':
            return im
        elif monochrome == 'MONOCHROME1':
            return 2**(n_bits-1) - im
                
                
def contours_to_csv(actors,names,filename):
    #Update save path
    path = os.path.join('./annotations',filename+'.csv')
    if not os.path.isdir('./annotations'):
        os.mkdir('./annotations')
    #Empty dictionary for ouput
    output_dict = {}
    #Iterate over contours and collect to output dictionary    
    for actor,name in zip(actors,names):
        if actor is not None:
            #Get points
            x,y = get_points_as_lists(actor)
            #Create temporary dictionary for the coordinates
            if name == 'Tibia, Lateral':
                _dict = {'TibiaLatX':x,'TibiaLatY':y}
            elif name == 'Tibia, Medial':
                _dict = {'TibiaMedX':x,'TibiaMedY':y}
            elif name == 'Femur, Lateral':
                _dict = {'FemurLatX':x,'FemurLatY':y}
            elif name == 'Femur, Medial':
                _dict = {'FemurMedX':x,'FemurMedY':y}
            #Update output dictionary
            output_dict.update(_dict)
    #Make sure all arrays have the same length, use NaN fill
    df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in output_dict.items()})        
    #Write csv
    df.to_csv(path, sep='\t', encoding='utf-8', index = False)
        
def csv_to_contour(filename):
    path = os.path.join('./annotations',filename)
    keys = ['TibiaLat','TibiaMed','FemurLat','FemurMed']
    names = ['Tibia, Lateral','Tibia, Medial', 'Femur, Lateral', 'Femur, Medial']
    actors = []
    df = pd.read_csv(path,sep='\t', encoding='utf-8')
    for key in keys:
        try:
            x = np.array(df[key+'X']).reshape(-1,1)
            y = np.array(df[key+'Y']).reshape(-1,1)
            nppoints = np.concatenate((x,y),1)
            actors.append(make_actor(nppoints))
        except KeyError:
            continue
    return actors,names

def csv_to_point_actor(filename,new_path=False):
    if new_path is True:
        path = filename
    else:
        path = os.path.join('.models','Points',filename)
    points = csv_to_points(path)
    actor = make_point_actor(points)
    return actor

def csv_to_clustered_arrays(filename,points_per_contour=20):
    '''Return 2 arrays containing tibial and femoral contours'''
    df = pd.read_csv(filename,sep='\t',encoding='utf-8')
    names = ['TibiaLat','TibiaMed','FemurLat','FemurMed']
    tibia = None; femur = None;
    for name in names:
        x = np.array(df[name+'X']); y = np.array(df[name+'Y']);
        #Remove nans
        x = x[~np.isnan(x)].reshape(-1,1); y = y[~np.isnan(y)].reshape(-1,1);
        _tmp = np.concatenate((x,y),1)
        #Fill gaps between the points
        _tmp = fill_points(_tmp)
        #20 point k-means clustering
        kmeans = KMeans(n_clusters=points_per_contour,random_state=42).fit(_tmp)
        _tmp = kmeans.cluster_centers_
        #Sort the cluster centers
        #Tibial contours
        if name == 'TibiaLat':
            _tmp = sort_points(_tmp,from_='ymax')
            tibia = _tmp
        elif name == 'TibiaMed':
            _tmp = sort_points(_tmp,from_='ymax')
            _tmp = deepcopy(np.flip(_tmp,axis=0))
            tibia = np.concatenate((tibia,_tmp),0)
        #Femoral contours
        elif name == 'FemurLat':
            _tmp = sort_points(_tmp,from_='ymin')
            femur = _tmp
        elif name == 'FemurMed':
            _tmp = sort_points(_tmp,from_='ymin')
            _tmp = deepcopy(np.flip(_tmp,axis=0))
            femur = np.concatenate((femur,_tmp),0)
    return tibia,femur

def points_to_csv(filename,points):
    _dict = {'x':points[:,0],'y':points[:,1]}
    df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in _dict.items()})        
    #Write csv
    df.to_csv(filename, sep='\t', encoding='utf-8', index = False)
    
def csv_to_points(filename):
    df = pd.read_csv(filename,sep='\t', encoding='utf-8')
    
    x = np.array(df['x']);y = np.array(df['y']);
    points = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),1)
    return points

def asm_to_csv(filename,mean_shape,principal_components,eigen_values,W):
    #Flatten the arrays to 1D vectors
    mu = mean_shape.squeeze()
    pc = principal_components.reshape(np.prod(principal_components.shape))
    val = eigen_values.squeeze()
    w = W.squeeze()
    _dict = {'mean_shape':mu,'principal_components':pc,'eigen_values':val,'W':w}
    #Convert to dataframe
    df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in _dict.items()})        
    #Write csv
    df.to_csv(filename, sep='\t', encoding='utf-8', index = False)
    
def csv_to_asm(filename):
    df = pd.read_csv(filename,sep='\t', encoding='utf-8')
    #Load data
    mu = np.array(df['mean_shape'])
    #Remove nan values and reshape
    mu = mu[~np.isnan(mu)].reshape(1,-1)
    val = np.array(df['eigen_values'])
    val = val[~np.isnan(val)]
    pc = np.array(df['principal_components'])
    pc = pc[~np.isnan(pc)].reshape(val.shape[0],pc.shape[0]//val.shape[0])
    w = np.array(df['W'])
    w = w[~np.isnan(w)]
    
    return mu,pc,val,w

def csv_to_roi(filename):
    #Read points and reshape
    df = pd.read_csv(filename,sep='\t', encoding='utf-8')
    sbRight = np.array(df['sbRight']);tRight = np.array(df['tRight']);
    sbLeft = np.array(df['sbLeft']);tLeft = np.array(df['tLeft']);
    sbRight = sbRight.reshape(sbRight.shape[0]//2,2);
    tRight = tRight.reshape(tRight.shape[0]//2,2);
    sbLeft = sbLeft.reshape(sbLeft.shape[0]//2,2);
    tLeft = tLeft.reshape(tLeft.shape[0]//2,2);
    #Append first point to the end of each shape
    sbRight = np.concatenate((sbRight,sbRight[0,:].reshape(1,-1)),0)
    tRight = np.concatenate((tRight,tRight[0,:].reshape(1,-1)),0)
    sbLeft = np.concatenate((sbLeft,sbLeft[0,:].reshape(1,-1)),0)
    tLeft = np.concatenate((tLeft,tLeft[0,:].reshape(1,-1)),0)
    
    #Make actor
    sbRactor = make_actor(sbRight);tRactor = make_actor(tRight);
    sbLactor = make_actor(sbLeft);tLactor = make_actor(tLeft);
    
    return sbRactor,tRactor,sbLactor,tLactor

def read_jsw(filename,choice='mean'):
    choices = ['min','max','mean','median']
    df = pd.read_csv(filename,sep='\t',encoding='utf-8')
    if choice == 'min':
        idx = 0
    elif choice == 'max':
        idx = 1
    elif choice == 'mean':
        idx = 2
    elif choice == 'median':
        idx == 3
    lateral = df['Lateral'][idx]
    medial = df['Medial'][idx]
    
    return lateral,medial

def read_js_edges(filename):
    edgefile = filename+'_edges.csv'
    df = pd.read_csv(edgefile,sep='\t',encoding='utf-8')
    lattib = np.array(df['tibia_lateral']);
    lattib = lattib.reshape(lattib.shape[0]//2,2)
    medtib = np.array(df['tibia_medial']);
    medtib = medtib.reshape(medtib.shape[0]//2,2)
    latfem = np.array(df['femur_lateral']);
    latfem = latfem.reshape(latfem.shape[0]//2,2)
    medfem = np.array(df['femur_medial']);
    medfem = medfem.reshape(medfem.shape[0]//2,2)
    
    #Make actors
    lattibActor = make_actor(lattib);medtibActor = make_actor(medtib);
    latfemActor = make_actor(latfem);medfemActor = make_actor(medfem);
    
    
    return lattibActor,medtibActor,latfemActor,medfemActor

def read_js_lines(filename):    
    pointsfile = filename+'_points.csv'
    df = pd.read_csv(pointsfile,sep='\t',encoding='utf-8')
    xlat = np.array(df['x_lateral']);ylat = np.array(df['y_lateral']);
    xmed = np.array(df['x_medial']);ymed = np.array(df['y_medial']);
    
    plateral = np.concatenate((xlat.reshape(-1,1),ylat.reshape(-1,1)),1)
    pmedial = np.concatenate((xmed.reshape(-1,1),ymed.reshape(-1,1)),1)
    
    Dlat = (plateral[0,:]-plateral[1,:])**2
    Dlat = Dlat.sum()**0.5*0.148
    Dmed = (pmedial[0,:]-pmedial[1,:])**2
    Dmed = Dmed.sum()**0.5*0.148
    
    platActor = make_actor(plateral);pmedActor=make_actor(pmedial)
    
    if plateral.sum() == 0:
        platActor = None; Dlat = None;
    if pmedial.sum() == 0:
        pmedActor = None; Dmed = None;
    
    return platActor,pmedActor,Dlat,Dmed
    
def read_jsw_mask(filename):
    imagepath = filename+'.png'
    locpath = filename+'.csv'
    im = cv2.imread(imagepath,cv2.IMREAD_COLOR)
    im = (im[:,:,1]+im[:,:,2])>0
    im = np.uint8(im*255)
    
    df = pd.read_csv(locpath,sep='\t',encoding='utf-8')
    
    x = df['x']; y = df['y'];
        
    return im,x,y

def save_manual_results(sample,path,js_lateral,js_medial,sbRight,tRight,sbLeft,tLeft):
    #Make directories
    if not os.path.isdir(path):
        os.mkdir(path)
    jspath = os.path.join(path,'JS_lines')
    if not os.path.isdir(jspath):
        os.mkdir(jspath)
    roipath = os.path.join(path,'ROIs')
    if not os.path.isdir(roipath):
        os.mkdir(roipath)
    
    if js_lateral is not None:    
        plat = get_line_ends(js_lateral)
    else:
        plat = np.zeros((2,2))
    
    if js_medial is not None:    
        pmed = get_line_ends(js_medial)
    else:
        pmed = np.zeros((2,2))
        
        
    #Save JSW lines
    dict_ = {'x_lateral':plat[:,0],'y_lateral':plat[:,1],'x_medial':pmed[:,0],'y_medial':pmed[:,1]}
    df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in dict_.items()})        
    #Write csv
    df.to_csv(os.path.join(jspath,sample+'_points.csv'), sep='\t', encoding='utf-8', index = False)
    
    #Save ROIs (exclude last point)
    if sbRight is not None:
        sbR = get_points(sbRight)[:-1,:2]
    else:
        sbR = np.zeros((4,2))
    if tRight is not None:
        tR = get_points(tRight)[:-1,:2]
    else:
        tR = np.zeros((4,2))
    if sbLeft is not None:
        sbL = get_points(sbLeft)[:-1,:2]
    else:
        sbL = np.zeros((4,2))
    if tLeft is not None:  
        tL = get_points(tLeft)[:-1,:2]
    else:
        tL = np.zeros((4,2))
    
    #Generate dataset
    dict_ = {'bounding_box':['x0','y0','x1', 'y0','x1', 'y1','x0', 'y1'],'sbRight':sbR.flatten(),'sbLeft':sbL.flatten(),
            'tRight':tR.flatten(),'tLeft':tL.flatten()}
    df = pd.DataFrame.from_dict({key:pd.Series(value) for key,value in dict_.items()})        
    #Write csv
    df.to_csv(os.path.join(roipath,sample+'.csv'), sep='\t', encoding='utf-8', index = False)
        
    return    
    