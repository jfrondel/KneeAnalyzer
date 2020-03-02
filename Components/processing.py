import numpy as np
import os
import cv2
import pandas as pd
import vtk
from vtk.util import numpy_support
from copy import deepcopy

def get_point_length(points):
    D = (points[0,:]-points[1,:])**2
    D = D.sum()**0.5
    return D

def get_points(vtkActor):
    #Get Polydata
    polydata = vtkActor.GetMapper().GetInputAsDataSet()
    #Get points
    points = polydata.GetPoints()
    #Iterate of the points, only get every second point
    new_points =[] 
    n_points = points.GetNumberOfPoints()
    for k in range(1,n_points,2):
        p = points.GetPoint(k)
        new_points.append(p)
        
    new_points = np.array(new_points)
        
    if new_points[0,1] < new_points[-1,1]:
        new_points = deepcopy(np.flip(new_points))
        
    return new_points

def get_line_ends(vtkActor):
    #Get Polydata
    polydata = vtkActor.GetMapper().GetInputAsDataSet()
    #Get points
    points = polydata.GetPoints()
    #Iterate of the points, only get every second point
    new_points =[] 
    n_points = points.GetNumberOfPoints()
    new_points.append(points.GetPoint(0))
    new_points.append(points.GetPoint(n_points-1))
        
    new_points = np.array(new_points)[:,:2]
        
    return new_points

#Get points from vtkPolyData
def get_points_as_lists(vtkActor):
    #Get Polydata
    polydata = vtkActor.GetMapper().GetInputAsDataSet()
    #Get points
    points = polydata.GetPoints()
    #Iterate of the points, only get every second point
    x=[]
    y = []
    n_points = points.GetNumberOfPoints()
    for k in range(1,n_points,2):
        p = points.GetPoint(k)
        x.append(p[0])
        y.append(p[1])
        
    return x,y

def make_actor(numpy_points):
    #Create components for points
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for k in range(numpy_points.shape[0]):
        x = numpy_points[k,0]
        y = numpy_points[k,1]
        if x is not 'NaN' and y is not 'NaN':
            if k == 0:
                id1 = points.InsertNextPoint([x,y,0])
                id2 = points.InsertNextPoint([x,y,0])
            else:
                id1 = points.InsertNextPoint(points.GetPoint(2*k-1))
                id2 = points.InsertNextPoint([x,y,0])
        
                lines.InsertNextCell(2)
                lines.InsertCellPoint(id1)
                lines.InsertCellPoint(id2)
            
    #Create new contour object
    contour = vtk.vtkPolyData()
    contour.SetPoints(points)
    contour.SetLines(lines)
    
    #Create new polydata mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(contour)
    #Create new vtk actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor

def make_point_actor(numpy_points):
    #Make vtk points and vertices
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    
    for point in numpy_points:
        id1 = points.InsertNextPoint([point[0],point[1],0])
        verts.InsertNextCell(1)
        verts.InsertCellPoint(id1)
        
    #Add points and vertices to polydata
    pixels = vtk.vtkPolyData()
    pixels.SetPoints(points)
    pixels.SetVerts(verts)
    
    #Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pixels)
    
    #Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor
    
        
def get_angle(vec1,vec2,mode = 'rad'):
    if vec1.shape is not vec2.shape:
        raise ValueError("Vector shapes [%s, %s] do not match!!" % (vec1.shape, vec2.shape))
    else:
        #Remove extra dimensions
        vec1 = vec1.squeeze()
        vec2 = vec2.squeeze()
        #Get lengths of the vectors
        abs1 = ((vec1**2).sum())**0.5
        abs2 = ((vec2**2).sum())**0.5
        #Get dot product and scale with lengths
        dot12 = 1/(abs1*abs2)*vec1.dot(vec2)
        #Get angle
        theta= np.arccos(dot12)
        if mode == 'rad':
            return theta
        elif mode == 'deg':
            return theta*180/np.pi

def fill_points(points):
    #Fills spaces between points
    filled = []
    n_points = points.shape[0]
    for k in range(1,n_points,1):
        #Get first and alst point
        start = points[k-1,:]
        stop = points[k,:]
        #Add fist point to the filled list
        filled.append(start)
        #Get difference between points
        diff = stop-start
        #Get distance between points as integer
        r = ((diff**2).sum())**0.5
        r = int(r)
        #Get unit length vector from the diference
        unit = diff/(r+1e-9)
        #Get angle between the points
        for step in range(r):
            new_point = step*unit+start
            filled.append(new_point)
    return np.array(filled)

def delete_row(array,index):
    rows,cols = array.shape
    output = None
    if rows > 1:
        output = np.zeros((rows-1,cols))
        for k in range(rows):
            if k<index:
                output[k,:] = array[k,:]
            elif k>index:
                output[k-1] = array[k,:]
            elif k == index:
                continue
    return output

def sort_points(points,from_='ymin'):
    choices = ['xmin','xmax','ymin','ymax']
    if from_ not in choices:
        raise ValueError("Wrong direction!! Choose direction from %s" % choices)
    #Get number of points and select first element
    _len = points.shape[0]
    _tmp = []
    if from_ == choices[0]:
        ind = np.argmin(points,axis=0)
        _tmp.append(points[ind[0],:])
        points = delete_row(points,ind[0])
    elif from_ == choices[1]:
        ind = np.argmax(points,axis=0)
        _tmp.append(points[ind[0],:])
        points = delete_row(points,ind[0])
    elif from_ == choices[2]:
        ind = np.argmin(points,axis=0)
        _tmp.append(points[ind[1],:])
        points = delete_row(points,ind[1])
    elif from_ == choices[3]:
        ind = np.argmax(points,axis=0)
        _tmp.append(points[ind[1],:])
        points = delete_row(points,ind[1])

    while len(_tmp)<_len:
        diff = (points-_tmp[-1])**2
        diff = diff.sum(1)
        ind = np.argmin(diff)
        _tmp.append(points[ind,:])
        points = delete_row(points,ind)
    _tmp = np.array(_tmp)
    
    return _tmp       
        
def cluster_points(points,n_points=80,n_iters=10,seed = 42):
    points = np.array(points[:,:2])
    N = points.shape[0]
    
    #Initialize centroids randomly from the points
    inds = np.linspace(0,N-1,n_points).astype(int)
    centroids = points[inds,:]
    #Keep end points
    centroids[0,:] = points[0,:]
    centroids[-1,:] = points[-1,:]
    
    #K-means clustering
    for iter in range(n_iters):
        dists = []
        for c in centroids:
            d = ((points-c)**2).sum(1)
            dists.append(d)
        dists = np.array(dists)
        inds = np.argmin(dists,0).squeeze()
        for k in range(centroids.shape[0]):
            N = (inds==k).sum()
            _points = points[inds==k,:]
            centroids[k,:] = _points.sum(0)/(N+1e-9)
        centroids[0,:] = points[0,:]
        centroids[-1,:] = points[-1,:]
        
    return centroids

def stratified_split(X,Y,split_size=0.25,seed=42):
    np.random.seed(seed)

    max_label = int(Y.max())
    min_label = int(Y.min())
    n_data = Y.shape[0]
    x,y,idx = [],[],[]
    for k in range(min_label,max_label+1,1):
        #Get current label instances
        inds = Y==k
        #Calculate split size from current labels
        n = int(inds.sum()*split_size)
        #Take instances of label k
        #Take random samples from data
        counter = 0
        while True:
            r = np.random.randint(0,n_data)
            if inds[r] and r not in idx and counter<n:
                idx.append(r)
                x.append(X[r])
                y.append(Y[r])
                counter += 1
            if counter == n:
                break

    return np.array(x),np.array(y),np.array(idx)
