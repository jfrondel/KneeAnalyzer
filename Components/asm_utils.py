import numpy as np
import cv2
from copy import deepcopy
from sklearn.decomposition import PCA
from .processing import delete_row
from .IO_utils import csv_to_asm, asm_to_csv
from .image_processing import grad_1d

def refine_joint_location(image,center,crop_size=120,resolution=0.148,prctile=40):
    #Crop image
    _size = crop_size/resolution
    x0 = int(center[0]-0.5*_size); x1 = int(x0+_size);
    y0 = int(center[1]-0.5*_size); y1 = int(y0+_size);
    
    crop = image[y0:y1,x0:x1]
    
    #Binarize cropped image
    t = np.percentile(crop,prctile)
    bw = (crop>t)*1
    
    #Find the edges of the knee
    idx1 = None; idx2 = None; w = 80;
    for k in range(bw.shape[1]):
        T1 = bw[:,k:k+w]; T2 = bw[:,-(k+w):-k];
        if T1.sum() > (w*bw.shape[0])*0.5 and idx1 == None:
            idx1 = k+w//2
        if T2.sum() > (w*bw.shape[0])*0.2 and idx2 == None:
            idx2 = bw.shape[1]-(k+w//2)
        if idx1 and idx2:
            break
            
    #Calculate x coordinate
    x = (idx2-idx1)//2+idx1
    
    #Find the y-coordinate for the joint center
    T = crop[:,x-20:x+20].sum(1)
    #Get gradient
    grad = grad_1d(T,r=15)
    
    y = np.argmax(np.abs(grad))
    
    scale = idx2-idx1
    
    return [x+x0,y+y0], scale

def find_normal(points,n_contours=1,connect=False):
    #Generate contours from the asm mean shape in case of discontinuous parts
    contours = []
    N = points.shape[0]//n_contours
    for k in range(n_contours):
        if k==n_contours-1:
            shape = points[k*N:,:]
        else:
            shape = points[k*N:(k+1)*N,:]
        shape = np.concatenate((shape[0,:].reshape(1,-1),shape,shape[-1,:].reshape(1,-1)),0)
        contours.append(shape)
    #Connect first and last shape if specified (used to compute contour normals)
    if connect:
        first = contours[0]; last = contours[-1];
        contours[0][0,:] = last[-1,:]
        contours[-1][-1,:] = first[0,:]



    #Find new points using contour normals
    for contour in contours:
        D = contour[2:,:]-contour[0:-2,:]
        L2 = np.sqrt((D**2).sum(1))
        unit_normal = np.zeros(D.shape)
        unit_normal[:,0],unit_normal[:,1] = -D[:,1]*1/(L2+1e-9),D[:,0]*1/(L2+1e-9)
        try:
            normal = np.concatenate((normal,unit_normal),0)
        except NameError:
            normal = unit_normal
            
    return normal

def find_grad_max(Dx,Dy,mag,coords,normals,R=100,orientation=[None]):
    #Make a list of gradient directions
    directions = []
    N = len(orientation)
    for k in range(N):
        L1 = L1 = (k-1)*N
        if k == N-1:
            L2 = coords.shape[0]
        else:
            L2 = (k+1)*coords.shape[0]//N
        for _ in range(L1,L2,1):
            directions.append(orientation[k])
    #Empty lists for output
    points,vals = [],[]
    h,w = mag.shape
    for k in range(-R,R,1):
        points_ = coords+k*normals
        points_ = points_.astype(int)
        vals_ = []
        for p,ori,cur in zip(points_,directions,coords):
            #Enforce dimensions of the image
            p[0] = np.max((p[0],0));p[0] = np.min((p[0],w-1));
            p[1] = np.max((p[1],0));p[1] = np.min((p[1],h-1));
            
            #Compute distance between the original position and suggested update
            dist = dist = (((p-cur)**2).sum())**0.5
            
            #Get gradient value, x,y and axes are in different order in points than in numpy
            
            #Case: no gradient direction specified, absolute value is used
            if ori == None:
                v_ = mag[p[1],p[0]]
            #Case: increasing gradient along horizontal (x - ) axis
            elif ori == '+dx':
                #v_ = np.sign(Dx[p[1],p[0]])*mag[p[1],p[0]]
                v_ = Dx[p[1],p[0]]                
            #Case: decreasing gradient along horizontal (x - ) axis
            elif ori == '-dx':
                #v_ = -1*np.sign(Dx[p[1],p[0]])*mag[p[1],p[0]]
                v_ = -1*Dx[p[1],p[0]]
            #Case: increasing gradient along vertical (y - ) axis
            elif ori == '+dy':
                #v_ = np.sign(Dy[p[1],p[0]])*mag[p[1],p[0]]
                v_ = Dy[p[1],p[0]]
            #Case: decreasing gradient along vertical (y - ) axis
            elif ori == '-dy':
                #v_ = -1*np.sign(Dy[p[1],p[0]])*mag[p[1],p[0]]
                v_ = -1*Dy[p[1],p[0]]
            
            if dist < 1:
                vals_.append(v_)
            else:
                vals_.append(v_*0.9+0.1*1/dist)
        vals.append(np.array(vals_))
        points.append(points_)
        
    #Convert to numpy array
    points = np.array(points); vals = np.array(vals)
    
    #Get gradient maxima
    inds = np.argmax(vals,0)
    output_points,output_vals = [],[]
    for k in range(len(inds)):
        idx = inds[k]
        output_points.append(points[idx,k,:])
        output_vals.append(vals[idx,k])
        
    return np.array(output_points),np.array(output_vals)

class model_size_scaler(object):
    def __init__(self,points):
        dists = np.zeros((2))
        for model in points:
            mu = model.mean(0)
            model -= mu
            mins = model.min(0)
            maxs = model.max(0)
            dists += maxs-mins
            
        self.scale = dists/len(points)
        
    def __call__(self,model,scale=None,enforce_range=False):
        if scale==None:
            mu = model.mean(0)
            model -= mu
            _min = model.min(0); _max = model.max(0)
            scale = _max-_min
            model /= scale
            model *= self.scale
            model += mu
        else:
            if enforce_range==False:
                mu = model.mean(0)
                model -= mu
                _min = model.min(0); _max = model.max(0)
                tmpscale = _max-_min
                model /= tmpscale
                model *= scale
                model += mu
            else:
                model = self.enforce_range(model, scale)
        
        return model
    
    def enforce_range(self,model,range=[0.9,1.1]):
        mu = model.mean(0)
        model -= mu
        _min = model.min(0); _max = model.max(0);
        scale = _max-_min
        
        tmpxmin = deepcopy(self.scale); tmpxmin[0] *= range[0];
        tmpxmax = deepcopy(self.scale); tmpxmax[0] *= range[1];
        tmpymin = deepcopy(self.scale); tmpymin[1] *= range[0];
        tmpymax = deepcopy(self.scale); tmpymax[1] *= range[1];
        
        if scale[0] < tmpxmin[0]:
            model /= scale
            model *= tmpxmin
        elif scale[0] > tmpxmax[0]:
            model /= scale
            model *= tmpxmax
        if scale[1] < tmpymin[1]:
            model /= scale
            model *= tmpymin
        elif scale[1] > tmpymax[1]:
            model /= scale
            model *= tmpymax
            
        model += mu
        
        return model
        
    
    def update_scale(self,scale,mode='width'):
        choices = ['new','width','height']
        if mode not in choices:
            raise ValueError("Incorrect mode!! Choose one from %s" % choices)
        if mode == choices[0]:
            self.scale = scale
        elif mode == choices[1]:
            factor = scale/self.scale[0]
            self.scale *= factor
        elif mode == choices[2]:
            factor = scale/self.scale[1]
            self.scale *= factor
    
def make_weights(points):
    variances = []
    for sample in points:
        V = []
        for k in range(sample.shape[0]):
            #Compute euclidean distance between current point and all the others
            cur_point = sample[k,:]
            other_points = delete_row(sample, k)
            D = (other_points-cur_point)**2
            D = D.sum(1)**0.5
            #Compute variance of the distances
            mu = D.sum()/D.shape[0]
            _var = ((D-mu)**2).sum()/D.shape[0]
            V.append(_var)
        variances.append(np.array(V))
    #Compute the sum of the variances across the models
    variances = np.array(variances)
    variances = variances.sum(0)
    variances = 1/(variances+1e-6)
    return variances
    
def procrustes_transform(points1,points2,W=None):
    #Create a weight matrix of ones if one is not given
    if W is None:
        W = np.ones(points1.shape[0])
    #Matrix elements for least squares approach
    x1 = W*points1[:,0]; y1 = W*points1[:,1];
    x2 = W*points2[:,0]; y2 = W*points2[:,1];
    z = W*(points2[:,0]**2+points2[:,1]**2);
    c1 = W*(points1[:,0]*points2[:,0]+points1[:,1]*points2[:,1]);
    c2 = W*(points1[:,1]*points2[:,0]-points1[:,0]*points2[:,1]);
    w = W.sum();
    
    x1 = x1.sum(); y1 = y1.sum(); x2 = x2.sum(); y2 = y2.sum();
    z = z.sum(); c1 = c1.sum(); c2 = c2.sum();
    
    #Solve coefficients for the aligment XA=Y
    X = np.array([[x2, -y2, w, 0],
                  [y2, x2, 0, w],
                  [z, 0, x2, y2],
                  [0, z, -y2, x2]])
    
    Y = np.array([x1,y1,c1,c2])
    
    X_inv = np.linalg.solve(X.T.dot(X),X.T)
    A = X_inv.dot(Y)
    
    #Apply transformation
    out = np.zeros(points2.shape)
    for k in range(points2.shape[0]):
        _x = points2[k,0]*A[0] - points2[k,1]*A[1] + A[2]
        _y = points2[k,0]*A[1] + points2[k,1]*A[0] + A[3]
        out[k,0] = _x; out[k,1] = _y;
    
    return out,A

def align_models(points,use_weights=False):
    if use_weights:
        W = make_weights(points)
    else:
        W = np.ones(points[0].shape[0])
    #Record mean scale along x and y axes from the original points
    scaler = model_size_scaler(deepcopy(points))
    
    #Align everything to first model
    aligned = [points[0]]
    for k in range(1,len(points)):
        _model,transformation = procrustes_transform(aligned[0],points[k],W)
        aligned.append(_model)
    aligned = np.array(aligned)
    #Compute mean shape
    mean_shape = scaler(aligned.mean(0))
    
    #Align all of the shapes to the mean shape and compute new mean shape until he model converges
    for k in range(50):
        aligned = []
        for model in points:
            _model, transformation = procrustes_transform(mean_shape,model,W)
            aligned.append(_model)
        aligned = np.array(aligned)
        new_mean_shape = scaler(aligned.mean(0))
        D = (mean_shape-new_mean_shape)**2
        D = D.sum()/mean_shape.shape[0]
        mean_shape = new_mean_shape
        if D<=1:
            #Align the shapes to the mean shape
            aligned = []
            for model in points:
                _model, transformation = procrustes_transform(mean_shape,model,W)
                aligned.append(_model)
            aligned = np.array(aligned)
            #Compute the final mean shape
            mean_shape = scaler(aligned.mean(0))
            #Flatten the shape vectors
            tmp = []
            for model in aligned:
                model = model.reshape(2*model.shape[0])
                tmp.append(model)
            aligned = np.array(tmp)
            mean_shape = mean_shape.reshape(1,2*mean_shape.shape[0])
            break
    
    return mean_shape,aligned,W,scaler

def get_principal_components(points,explained_variance = 0.97):
    _pca = PCA(svd_solver='full',random_state=42,copy=True,whiten=False).fit(points)
    comps= _pca.components_
    svals = _pca.singular_values_
    vars_ = _pca.explained_variance_ratio_
    n = 0
    total_v = 0
    for v in vars_:
        n += 1
        total_v += v
        if total_v >= explained_variance:
            break
    return comps[:n,:],svals[:n]
        
class ASM(object):
    def __init__(self,points,use_weights=True,train=True):

        self.use_weights = use_weights
        self.scaler = None
        if train:        
            self.train(points,use_weights)

        return

    def __call__(self,points,size_range=[0.9,1.1]):
        #Deforms the active shape model to the given set of points

        #Aligne the mean shape and the points
        mu = self.mean_shape.reshape(points.shape)
        pc = self.components.astype(np.float32)
        aligned,_ = procrustes_transform(points,mu,self.W)

        #Compute residual transformation
        dx = (points-aligned)
        dx = dx.reshape(2*dx.shape[0],1)

        #Enforce limits to the residual transformtation
        b = pc.dot(dx)
        for k in range(b.shape[0]):
            #Get the limit from the corresponding eigenvalue
            lim_ = 2*np.abs(self.vals[k])**0.5
            if np.abs(b[k]) > lim_:
                b[k] = np.sign(b[k])*lim_

        #Compute the updated residual transformation
        b = b.reshape(1,-1)
        dx_new = np.matmul(b,self.components).squeeze()
        dx_new = dx_new.reshape(dx_new.shape[0]//2,2)
        dx_new += aligned
        
        #Enforce scale
        #dx_new = self.scaler(dx_new,scale=size_range,enforce_range=True)
        dx_new = self.scaler(dx_new,scale=None)

        return dx_new

    def train(self,points,use_weights):
        #Align the points and compute mean shape
        self.mean_shape,self.aligned,self.W,self.scaler = align_models(points,use_weights)
        self.aligned = np.array(self.aligned)
        #Get principal components and eigenvalues
        self.components,self.vals = get_principal_components(self.aligned-self.mean_shape)
        
        return
    
    def update_scale(self,scale,mode='width'):
        self.scaler.update_scale(scale,mode=mode)    
    
    def load(self,file):
        self.mean_shape,self.components,self.vals,self.W = csv_to_asm(file)
        mu = self.mean_shape.reshape(self.mean_shape.shape[1]//2,2)
        self.scaler = model_size_scaler([mu])
        
    def save(self,file):
        asm_to_csv(file, self.mean_shape, self.components, self.vals, self.W)
        