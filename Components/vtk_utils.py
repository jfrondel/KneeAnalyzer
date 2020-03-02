import numpy as np
import vtk
from vtk.util import numpy_support

#from .interactor_utils import image_interactor, draw_interactor, draw_line
from .interactor_utils2 import image_interactor,draw_interactor
from copy import deepcopy

def numpy_to_vtk(imagedata,normalize=True,start=None):
    #Rescale  grayscale values to range [0,1]
    if normalize == True:
        _min,_max = imagedata.min(),imagedata.max()
        imagedata = imagedata-_min
        imagedata = imagedata/_max
        #Convert to 8 bit grayscale image
        imagedata = np.uint8(imagedata*255)
    else:
        imagedata = np.uint8(imagedata)
    
    #Get dimensions
    dims = imagedata.shape
        
    #Convert to vtkdata
    vtkdata = numpy_support.numpy_to_vtk(num_array=imagedata.ravel(),deep=True,array_type=vtk.VTK_TYPE_UINT8)
    
    #Create vtk image
    vtkimage = vtk.vtkImageData()
    if len(dims)==2:
        vtkimage.SetDimensions(dims[1],dims[0],1)
        if start is None:
            vtkimage.SetExtent(0,dims[1]-1,0,dims[0]-1,0,0)
        else:
            vtkimage.SetExtent(start[0],start[0]+dims[1]-1,start[1],start[1]+dims[0]-1,0,0)
    elif len(dims)==3:
        vtkimage.SetDimensions(dims[1],dims[0],dims[2])
        if start is None:
            vtkimage.SetExtent(0,dims[1]-1,0,dims[0]-1,0,dims[2]-1)
        else:
            vtkimage.SetExtent(start[1],start[1]+dims[1]-1,start[0],start[0]+dims[0]-1,0,dims[2]-1)
    vtkimage.SetSpacing([1,1,1])
    vtkimage.SetOrigin([0,0,0])
    vtkimage.GetPointData().SetScalars(vtkdata)
    
    return vtkimage

def set_image_color(vtkimagedata,max_value,color):
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(0, max_value)
    for k in range(max_value):
        val = k/(max_value-1)
        lut.SetTableValue(k,val*color[0],val*color[1],val*color[2],val*color[3])
    lut.Build()
    
    colormapper = vtk.vtkImageMapToColors()
    colormapper.SetLookupTable(lut)
    colormapper.SetInputData(vtkimagedata)
    
    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputConnection(colormapper.GetOutputPort())
    
    return actor

def numpy_to_actor2d(image,normalize=True,start=None,color=None,max_color=256):
    vtkimage = numpy_to_vtk(image,normalize,start)
    if color is None:
        actor = vtk.vtkImageActor()
        actor.SetInputData(vtkimage)
    else:
        actor = set_image_color(vtkimage,max_color,color)
    return actor

def set2d_camera(renderer):
    cam = vtk.vtkCamera()
    cam.SetViewUp(0,-1,0)
    pos = cam.GetPosition()
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToDisplay()
    coord.SetValue(pos[0], pos[1], pos[2])
    newpos = coord.GetComputedWorldValue(renderer)
    cam.SetPosition(newpos[0], newpos[1], -newpos[2])
    foc = cam.GetFocalPoint()
    focCoord = vtk.vtkCoordinate()
    focCoord.SetCoordinateSystemToDisplay()
    focCoord.SetValue(foc[0], foc[1], foc[2])
    newFoc = focCoord.GetComputedWorldValue(renderer)
    cam.SetFocalPoint(newFoc[0], newFoc[1], -1)
    cam.SetClippingRange(1e-5, 1)
    renderer.ResetCamera()        
        
def make_polyDataActor(points,mode='lines'):
    choices = ['lines','verts']
    if not mode in choices:
        raise ValueError('Choose valid mode from %s' % choices)
    else:
        #New poly data
        polyData = vtk.vtkPolyData()
        #Create points
        vtkPoints = vtk.vtkPoints()
        vtkCells = vtk.vtkCellArray()
        for k in range(points.shape[0]):
            x = points[k,0]
            y = points[k,1]
            if mode == choices[0]:
                if k == 0:
                    id1 = vtkPoints.InsertNextPoint([x, y, 0])
                    id2 = vtkPoints.InsertNextPoint([x, y, 0])
                else:
                    n_points = vtkPoints.GetNumberOfPoints()
                    id1 = vtk.vtkPoints.InsertNextPoint(vtkPoints.GetPoint(n_points))
                    id2 = vtkPoints.InsertNextPoint([x, y, 0])
                vtkCells.InsertNextCell(2)
                vtkCells.InsertCellPoint(id1)
                vtkCells.InsertCellPoint(id2)
                
            elif mode == choices[1]:
                x = points[k,0]
                y = points[k,1]
                id = vtkPoints.InsertNextPoint([x,y,0])
                vtkCells.InsertNextCell(1)
                vtkCells.InsertCellPoint(id)
        
        polyData.SetPoints(vtkPoints)
        if mode == choices[0]:
            polyData.SetLines(vtkCells)
        elif mode == choices[1]:
            polyData.SetVerts(vtkCells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

def points_to_polyData(points,mode='lines'):
    vtkPoints = vtk.vtkPointData()
    vtkCells = vtk.vtkCellArray()
    

class text_actor(object):
    def __init__(self,color = [0,1,1], font_size = 16, position = [10,10]):
        self.txt = None
        self.color = color
        self.font_size = font_size
        self.position = position
        
    def render_text(self,renWin, input_text):
        #Get renderer and remove old actor
        renderer = renWin.GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.txt)
        
        #Update the current actor
        self.txt = vtk.vtkTextActor()
        self.txt.SetInput(input_text)
        #Update the text properties
        self.txt.GetTextProperty().SetFontFamilyToArial()
        self.txt.GetTextProperty().SetFontSize(self.font_size)
        self.txt.GetTextProperty().SetColor(self.color[0],self.color[1],self.color[2])
        #Update position
        self.txt.SetDisplayPosition(self.position[0], self.position[1])
        
        renderer.AddActor(self.txt)
        renderer.Modified()
        renWin.Render()
        
        return
        
    def clear_text(self,renWin):
         #Get renderer and remove old actor
        renderer = renWin.GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.txt)
        self.txt = None
        return
    
class mask_Actor(object):    
    def __init__(self,color=[1.0,1.0,0.0,1.0]):
        self.color = color
        self.actor = None
        
    def render_mask(self,renWin,mask,x,y):
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            start = np.array([x,y])
            self.actor = numpy_to_actor2d(mask,normalize=True,start=start,color=self.color)
            renderer.AddActor(self.actor)
            renderer.Modified()
            renWin.Render()
            
    def clear_mask(self,renWin):
        renderer = renWin.GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.actor)
        self.actor = None
        renderer.Modified()
        renWin.Render()
        
        

class xray_rendering(object):
    def __init__(self,vtkWidget,functions_=[None,None]):
        self.vtkWidget = vtkWidget
        self.renderer = None
        self.imageActor = None
        
        #Text actor
        self.txt = text_actor()
        #Mask actor
        self.maskActor = mask_Actor(color=[1.0,1.0,0.0,0.2])
        
        #Interator styles
        self.moveStyle = image_interactor()
        self.drawStyle = draw_interactor()
        self.lineStyle = draw_interactor(mode='line',func_=functions_[0])
        self.boxStyleSmall = draw_interactor(mode='box',h=41,w=95,func_=functions_[1])
        self.boxStyleBig = draw_interactor(mode='box',h=95,w=95,func_=functions_[1])
        #self.lineStyle = draw_line()
        
    def render_image(self,image):
        #Get render window, renderer and interactor
        renWin = self.vtkWidget.GetRenderWindow()
        iren = renWin.GetInteractor()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        
        #Remove old components
        if renderer is not None:
            renderer.RemoveActor(self.imageActor)
            renWin.RemoveRenderer(renderer)
        else:
            renderer = vtk.vtkRenderer()
            
        #Create image actor
        self.imageActor = numpy_to_actor2d(image)
        
        #Connect components to the rendering object
        renderer.AddActor(self.imageActor)
        renWin.AddRenderer(renderer)
        #Set camera view up
        cam = renderer.GetActiveCamera()
        campos = cam.GetPosition()
        cam.SetPosition(campos[0],campos[1],-campos[2])
        cam.SetViewUp(0,-1,0)
        renWin.GetRenderers().GetFirstRenderer().ResetCamera()
        
        #Update interactor style
        if iren is None:
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)
        #iren.SetInteractorStyle(self.moveStyle)
        self.set_interactor()
        renWin.Render()
        iren.Start()
        
    def rendermask(self,mask,x,y):
        self.maskActor.render_mask(self.vtkWidget.GetRenderWindow(), mask, x, y)
        
    def clearmask(self):
        self.maskActor.clear_mask(self.vtkWidget.GetRenderWindow())       
        
    def render_text(self,input_text):
        self.txt.render_text(self.vtkWidget.GetRenderWindow(), input_text)
        
    def set_interactor(self,interactor='image'):
        choices = ['image','draw','line','box_small','box_big']
        if interactor not in choices:
            raise ValueError('Choose valid interactor from following: %s' % choices)
        else:
            renWin = self.vtkWidget.GetRenderWindow()
            iren = renWin.GetInteractor()
            if iren is None:
                iren = vtk.vtkRenderWindowInteractor()
                iren.SetRenderWindow(renWin)
            if interactor == choices[0]:
                iren.SetInteractorStyle(self.moveStyle)
            elif interactor == choices[1]:
                iren.SetInteractorStyle(self.drawStyle)
            elif interactor == choices[2]:
                iren.SetInteractorStyle(self.lineStyle)
            elif interactor == choices[3]:
                iren.SetInteractorStyle(self.boxStyleSmall)
            elif interactor == choices[4]:
                iren.SetInteractorStyle(self.boxStyleBig)
            
        
    def clear_actors(self):
        #Get render window, renderer and interactor
        renWin = self.vtkWidget.GetRenderWindow()
        iren = renWin.GetInteractor()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old components
        if renderer is not None:
            renderer.RemoveActor(self.imageActor)
            self.txt.clear_text(renWin)
            self.maskActor.clear_mask(renWin)
            if iren is not None:
                if type(iren.GetInteractorStyle()) == type(self.drawStyle):
                    iren.GetInteractorStyle().clear_actor()
            renWin.RemoveRenderer(renderer)
            
    def clear_draw_actor(self):
        #Get render window, renderer and interactor
        renWin = self.vtkWidget.GetRenderWindow()
        iren = renWin.GetInteractor()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old components
        if renderer is not None:
            if iren is not None:
                if type(iren.GetInteractorStyle()) == type(self.drawStyle):
                    iren.GetInteractorStyle().clear_actor()
            
    def get_draw_actor(self):
        actor = None
        #Get render window, renderer and interactor
        renWin = self.vtkWidget.GetRenderWindow()
        iren = renWin.GetInteractor()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old components
        if renderer is not None:
            if iren is not None:
                if type(iren.GetInteractorStyle()) is not type(self.moveStyle):
                    actor = iren.GetInteractorStyle().get_contour_components()
                    
        return actor