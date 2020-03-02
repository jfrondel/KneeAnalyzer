import numpy as np
import os
import vtk
import gc
from copy import deepcopy
from .processing import make_actor
#from .vtk_utils import text_actor

def get_coordinate(vtkinteractor,mode='world'):
    choices = ['world','display']
    if mode not in choices:
        raise ValueError("Invalid mode, select one from %s" % choices)
    #Get event position
    pos = vtkinteractor.GetEventPosition()
    
    #Get renderer
    renWin = vtkinteractor.GetRenderWindow()
    renderer = renWin.GetRenderers().GetFirstRenderer()
    
    #Compute coordinate
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToDisplay()
    coord.SetValue(pos[0], pos[1], 0)
    if mode == 'world':
        output = coord.GetComputedWorldValue(renderer)
    elif mode == 'display':
        output = coord.GetComputedDisplayValue(renderer)
        
    return output
   
def get_camera_coordinate(renderer,mode='world'):
    choices = ['world','display']
    if mode not in choices:
        raise ValueError("Invalid mode, select one from %s" % choices)    
    
    #Get camera
    cam = renderer.GetActiveCamera()
    
    #Get focal point
    foc_ = cam.GetFocalPoint()
    foc_coord = vtk.vtkCoordinate()
    foc_coord.SetCoordinateSystemToDisplay()
    foc_coord.SetValue(foc_[0], foc_[1], foc_[2])
    
    #Get position
    pos_ = cam.GetPosition()
    pos_coord = vtk.vtkCoordinate()
    pos_coord.SetCoordinateSystemToDisplay()
    pos_coord.SetValue(pos_[0], pos_[1], pos_[2])
    
    if mode == 'world':
        foc = foc_coord.GetComputedWorldValue(renderer)
        pos = pos_coord.GetComputedWorldValue(renderer)
    elif mode == 'display':
        foc = foc_coord.GetComputedDisplayValue(renderer)
        pos = pos_coord.GetComputedDisplayValue(renderer)
    
    return pos,foc

class contour(object):
    def __init__(self,width=3,color=[1.0,0.0,0.0]):
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.polydata = vtk.vtkPolyData()
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        
        self.width = width
        self.color = color
        
        return
    
    def append(self,point):
        #Delete old actor and mapper
        self.actor = None
        self.mapper = None
        #Get number of points
        N = self.points.GetNumberOfPoints()
        #Insert new points
        if N == 0:
            id1 = self.points.InsertNextPoint([point[0],point[1],0])
            id2 = self.points.InsertNextPoint([point[0],point[1],0])
        elif N > 0:
            tmp = self.points.GetPoint(N-1)
            id1 = self.points.InsertNextPoint(tmp)
            id2 = self.points.InsertNextPoint([point[0],point[1],0])
        #Insert new line
        self.lines.InsertNextCell(2)
        self.lines.InsertCellPoint(id1)
        self.lines.InsertCellPoint(id2)
        #Update polydata
        self.polydata.SetPoints(self.points)
        self.polydata.SetLines(self.lines)
        self.polydata.Modified()
        #Create new actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.polydata)
        
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        
        #Update actor properties
        self.actor.GetProperty().SetLineWidth(self.width)
        self.actor.GetProperty().SetColor(self.color)
        
        return self.actor
        
    def new_points(self,points):
        #Delete old actor and mapper
        self.clear()
        
        #Iterate over points
        for p in points:
            #Get number of points
            N = self.points.GetNumberOfPoints()
            #Insert new points
            if N == 0:
                id1 = self.points.InsertNextPoint([p[0],p[1],0])
                id2 = self.points.InsertNextPoint([p[0],p[1],0])
            elif N > 0:
                tmp = self.points.GetPoint(N-1)
                id1 = self.points.InsertNextPoint(tmp)
                id2 = self.points.InsertNextPoint([p[0],p[1],0])
            #Insert new line
            self.lines.InsertNextCell(2)
            self.lines.InsertCellPoint(id1)
            self.lines.InsertCellPoint(id2)
            
        #Update polydata
        self.polydata.SetPoints(self.points)
        self.polydata.SetLines(self.lines)
        self.polydata.Modified()
        #Create new actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.polydata)
        
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        
        #Update actor properties
        self.actor.GetProperty().SetLineWidth(self.width)
        self.actor.GetProperty().SetColor(self.color)
        
        return self.actor
            
    def clear(self):
        #Set old components to none
        self.actor = None
        self.mapper = None
        self.points = None
        self.lines = None
        self.polydata = None
        gc.collect()
        
        #Create new objects
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.polydata = vtk.vtkPolyData()
        
class cursor(object):
    def __init__(self,interactorstyle,color=[1.0,0.0,0.0],R=5):
        self.interactorstyle = interactorstyle
        self.line1 = contour()
        self.line2 = contour()
        self.actor1 = None;
        self.actor2 = None;
        self.lw = 3
        self.color = color
        self.R = R
        return
    
    def __move__(self):
        #Get components
        interactor = self.interactorstyle.GetInteractor() 
        renWin = interactor.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old actor
        renderer.RemoveActor(self.actor1);renderer.RemoveActor(self.actor2);
        
        #Get center
        center = get_coordinate(interactor, mode='world')
        #Create actors
        points1 = np.array([[center[0]-self.R,center[1]-self.R],[center[0]+self.R,center[1]+self.R]])
        points2 = np.array([[center[0]-self.R,center[1]+self.R],[center[0]+self.R,center[1]-self.R]])
        self.actor1 = self.line1.new_points(points1);self.actor2 = self.line2.new_points(points2);
        

        renderer.AddActor(self.actor1);renderer.AddActor(self.actor2);
        renderer.Modified()
        renWin.Render()
        
        return
    
    def remove_cursor(self):
        #Get components
        interactor = self.interactorstyle.GetInteractor() 
        renWin = interactor.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old actor
        renderer.RemoveActor(self.actor1);
        renderer.RemoveActor(self.actor2);
        renderer.Modified()
        renWin.Render()
        
        return

class box_cursor(object):
    def __init__(self,interactorstyle,line_width=3,color=[1.0,0.0,0.0],h=41,w=95):
        self.interactorstyle = interactorstyle
        self.box = contour()
        self.actor = None;
        self.lw = line_width
        self.color = color
        self.h = h
        self.w = w
        return
    
    def __move__(self):
                #Get components
        interactor = self.interactorstyle.GetInteractor() 
        renWin = interactor.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old actor
        renderer.RemoveActor(self.actor);
        self.actor = None
        
        #Get center
        center = get_coordinate(interactor, mode='world')
        
        points = np.array([[center[0]-self.w//2,center[1]-self.h//2],
                           [center[0]+self.w//2,center[1]-self.h//2],
                           [center[0]+self.w//2,center[1]+self.h//2],
                           [center[0]-self.w//2,center[1]+self.h//2],
                           [center[0]-self.w//2,center[1]-self.h//2]])
        
        self.actor = self.box.new_points(points)
        
        renderer.AddActor(self.actor)
        renderer.Modified()
        
        renWin.Render()
        
        return
        
    def remove_cursor(self):
        #Get components
        interactor = self.interactorstyle.GetInteractor() 
        renWin = interactor.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old actor
        renderer.RemoveActor(self.actor);
        renderer.Modified()
        renWin.Render()
        
        return
        
class image_interactor(vtk.vtkInteractorStyle):
    def __init__(self):
        self.AddObserver('LeftButtonPressEvent',self.leftDownEvt)
        self.AddObserver('LeftButtonReleaseEvent',self.leftUpEvt)
        self.AddObserver('MouseMoveEvent',self.Move2d)
        self.AddObserver('MouseWheelForwardEvent',self.wheelFrwdEvt)
        self.AddObserver('MouseWheelBackwardEvent',self.wheelBkwdEvt)
        self.click = [0,0,0]
        self.pos = [0,0,0]
        self.foc = [0,0,0]
        
        self.leftDown = False
        
    def leftDownEvt(self,obj,event):
        if self.leftDown == False:
            #Get current camera position
            interactor = self.GetInteractor()
            renWin = interactor.GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            
            #Get click location and compute world coordinates
            clickPos = get_coordinate(interactor, mode='world')
            
            #Get camera position and focal point
            pos,foc = get_camera_coordinate(renderer, mode='world')
            
            #Update variables
            self.click = [clickPos[0],clickPos[1],clickPos[2]]
            self.pos = [pos[0],pos[1],pos[2]]
            self.foc = [foc[0],foc[1],0]
            self.leftDown = True

        return
    
    def leftUpEvt(self,obj,event):
        if self.leftDown == True:
            self.leftDown = False
        return
    
    def Move2d(self,obj,event):        
        if self.leftDown == True:
            #get interactor
            interactor = self.GetInteractor()
            #Get event position
            renWin = interactor.GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            move = self.GetInteractor().GetEventPosition()
            #Compute position in world coordinates
            pos = get_coordinate(interactor, mode='world')
            #Compute new location, opposite of the movement direction
            self.pos = [self.pos[0]-(pos[0]-self.click[0]),
                        self.pos[1]-(pos[1]-self.click[1]),
                        self.pos[2]]
            
            cam = renderer.GetActiveCamera()
            cam.SetPosition(self.pos[0],self.pos[1],self.pos[2])
            cam.SetFocalPoint(self.pos[0],self.pos[1],self.foc[2])
            renWin.Render()
            
        return
    
    def wheelFrwdEvt(self,obj,event):
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        cam = renderer.GetActiveCamera()
        cam.Zoom(1.1)
        renWin.Render()
        return
    
    def wheelBkwdEvt(self,obj,event):
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        cam = renderer.GetActiveCamera()
        cam.Zoom(0.9)
        renWin.Render()
        return

class draw_interactor(vtk.vtkInteractorStyle):
    def __init__(self,line_width=3,line_color=[1.0,0.0,0.0],mode='contour',h=41,w=95,func_=None):
        choices = ['contour','line','box']
        if mode not in choices:
            raise ValueError("Invalid mode, select one from: %s" % choices)
        self.AddObserver('LeftButtonPressEvent',self.leftDownEvt)
        self.AddObserver('LeftButtonReleaseEvent',self.leftUpEvt)
        self.AddObserver('RightButtonPressEvent',self.rightDownEvt)
        self.AddObserver('RightButtonReleaseEvent',self.rightUpEvt)
        self.AddObserver('MouseMoveEvent',self.Move2d)
        self.AddObserver('MouseWheelForwardEvent',self.wheelFrwdEvt)
        self.AddObserver('MouseWheelBackwardEvent',self.wheelBkwdEvt)
                
        
        self.leftDown = False
        self.rightDown = False
        
        #Mode
        self.mode = mode
        
        #Function to call
        self.func = func_
        
        #Line properties
        self.lw = line_width
        self.lc = line_color
        
        self.h = h
        self.w = w
        
        self.init_components()
        
        if mode is not 'box':
            self.cursor = cursor(self)
        else:
            self.cursor = box_cursor(self,h=self.h,w=self.w,color=self.lc,line_width=self.lw)
        
        return
    
    def init_components(self):
        self.click = [0,0,0]
        self.pos = [0,0,0]
        self.foc = [0,0,0]
        
        self.points = np.zeros((2,2))
        
        self.contour = contour(width=self.lw,color=self.lc)
        
        self.actor = None
        
      
    def leftDownEvt(self,obj,event):
        if self.leftDown == False:
            if self.mode is 'contour':
                #Get interactor
                interactor = self.GetInteractor()
                #Get render window  components
                renWin = interactor.GetRenderWindow()
                renderer = renWin.GetRenderers().GetFirstRenderer()
                #Remove old contour actor
                renderer.RemoveActor(self.actor)
                self.actor = None
                #Compute click coordinates
                pos = get_coordinate(interactor, mode='world')
                point = np.array([pos[0],pos[1]])
                
                self.actor = self.contour.append(point)
                
                renderer.AddActor(self.actor)
                renWin.Render()
                
            elif self.mode is 'line':
                #Get interactor
                interactor = self.GetInteractor()
                #Get render window  components
                renWin = interactor.GetRenderWindow()
                renderer = renWin.GetRenderers().GetFirstRenderer()
                #Remove old contour actor
                renderer.RemoveActor(self.actor)
                self.actor = None
                #Compute click coordinates
                pos = get_coordinate(interactor, mode='world')
                point = np.array([pos[0],pos[1]])
                self.points[0,:] = point;self.points[1,:] = point;
                
                self.actor = self.contour.new_points(self.points)
                
                renderer.AddActor(self.actor)
                renWin.Render()
                
            else:
                pass
            
            self.leftDown = True
            
        return
    
    def leftUpEvt(self,obj,event):
        if self.leftDown == True:
            if self.mode is not 'contour':
                self.func()
            self.leftDown = False
        return  
        
    def rightDownEvt(self,obj,event):
        if self.rightDown == False:
            #Get current camera position
            interactor = self.GetInteractor()
            renWin = interactor.GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            
            #Get click location and compute world coordinates
            clickPos = get_coordinate(interactor, mode='world')
            
            #Get camera position and focal point
            pos,foc = get_camera_coordinate(renderer, mode='world')
            
            #Update variables
            self.click = [clickPos[0],clickPos[1],clickPos[2]]
            self.pos = [pos[0],pos[1],pos[2]]
            self.foc = [foc[0],foc[1],0]
            self.rightDown = True

        return
    
    def rightUpEvt(self,obj,event):
        if self.rightDown == True:
            self.rightDown = False
        return
    
    def Move2d(self,obj,event):
        self.cursor.__move__()
        if self.leftDown == True:
            if self.mode is 'contour':
                #Get interactor
                interactor = self.GetInteractor()
                #Get render window  components
                renWin = interactor.GetRenderWindow()
                renderer = renWin.GetRenderers().GetFirstRenderer()
                #Remove old contour actor
                renderer.RemoveActor(self.actor)
                self.actor = None
                #Compute click coordinates
                pos = get_coordinate(interactor, mode='world')
                point = np.array([pos[0],pos[1]])
                
                self.actor = self.contour.append(point)
                
                renderer.AddActor(self.actor)
                renWin.Render()
                
            elif self.mode is 'line':
                                #Get interactor
                interactor = self.GetInteractor()
                #Get render window  components
                renWin = interactor.GetRenderWindow()
                renderer = renWin.GetRenderers().GetFirstRenderer()
                #Remove old contour actor
                renderer.RemoveActor(self.actor)
                self.actor = None
                #Compute click coordinates
                pos = get_coordinate(interactor, mode='world')
                point = np.array([pos[0],pos[1]])
                self.points[1,:] = point
                
                self.actor = self.contour.new_points(self.points)
                
                renderer.AddActor(self.actor)
                renWin.Render()
            
            
        if self.rightDown == True:
            #get interactor
            interactor = self.GetInteractor()
            #Get event position
            renWin = interactor.GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            move = self.GetInteractor().GetEventPosition()
            #Compute position in world coordinates
            pos = get_coordinate(interactor, mode='world')
            #Compute new location, opposite of the movement direction
            self.pos = [self.pos[0]-(pos[0]-self.click[0]),
                        self.pos[1]-(pos[1]-self.click[1]),
                        self.pos[2]]
            
            cam = renderer.GetActiveCamera()
            cam.SetPosition(self.pos[0],self.pos[1],self.pos[2])
            cam.SetFocalPoint(self.pos[0],self.pos[1],self.foc[2])
            renWin.Render()
            
        return
    
    def wheelFrwdEvt(self,obj,event):
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        cam = renderer.GetActiveCamera()
        cam.Zoom(1.1)
        renWin.Render()
        return
    
    def wheelBkwdEvt(self,obj,event):
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        cam = renderer.GetActiveCamera()
        cam.Zoom(0.9)
        renWin.Render()
        return
    
    def clear_actor(self):
        #Remove actor from renderwindow
        self.cursor.remove_cursor()
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.actor)
        renderer.Modified()
        #Reset points/lines
        self.init_components()
        renWin.Render()
        
    def get_contour_components(self):
        if self.mode is not 'box':
            actor_copy = vtk.vtkActor()
            actor_copy.ShallowCopy(self.actor)
        else:
            actor_copy = vtk.vtkActor()
            actor_copy.ShallowCopy(self.cursor.actor)        
        return actor_copy

'''    
class draw_line(vtk.vtkInteractorStyle):
    def __init__(self,line_width=3,line_color=[1.0,0.0,0.0],func_):
        self.AddObserver('LeftButtonPressEvent',self.leftDownEvt)
        self.AddObserver('LeftButtonReleaseEvent',self.leftUpEvt)
        self.AddObserver('RightButtonPressEvent',self.rightDownEvt)
        self.AddObserver('RightButtonReleaseEvent',self.rightUpEvt)
        self.AddObserver('MouseMoveEvent',self.Move2d)
        self.AddObserver('MouseWheelForwardEvent',self.wheelFrwdEvt)
        self.AddObserver('MouseWheelBackwardEvent',self.wheelBkwdEvt)
                
        
        self.leftDown = False
        self.rightDown = False
        
        #Line properties
        self.lw = line_width
        self.lc = line_color
        
        #Empty actor
        self.actor = None
        
        self.initialize()
        
        self.cursor = cursor(self)
        
        self.func = func_
        
    def initialize(self):
        self.click = [0,0,0]
        self.pos = [0,0,0]
        self.foc = [0,0,0]
        self.points = np.zeros((2,2))
        self.jsw = 0.0
        
      
    def leftDownEvt(self,obj,event):
        if self.leftDown == False:
            #Get render window  components
            renWin = self.GetInteractor().GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            #Remove old contour actor
            renderer.RemoveActor(self.actor)
            self.actor = None
            #Compute click coordinates
            click = self.GetInteractor().GetEventPosition()            
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(click[0], click[1], 0)
            clickPos = coord.GetComputedWorldValue(renderer)
            #Record point
            self.points[0,:] = np.array([clickPos[0],clickPos[1]])
            self.points[1,:] = np.array([clickPos[0],clickPos[1]])
            
            self.actor = make_actor(self.points)
            
            self.actor.GetProperty().SetLineWidth(self.lw)
            self.actor.GetProperty().SetColor(self.lc)
            #Update actor position
            actPos = self.actor.GetPosition()
            self.actor.SetPosition(actPos[0], actPos[1], -1)
            
            renderer.AddActor(self.actor)
            renderer.Modified()
            renWin.Render()
            
            jsw = (self.points[0,:]-self.points[1,:])**2
            jsw = (jsw.sum()**0.5)*0.148
            self.jsw = jsw
            
            self.leftDown = True
            
        return
    
    def leftUpEvt(self,obj,event):
        if self.leftDown == True:
            self.leftDown = False
            self.func()
        return  
        
    def rightDownEvt(self,obj,event):
        if self.rightDown == False:
            #Get current camera position
            renWin = self.GetInteractor().GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            #Get click location and compute world coordinates
            click = self.GetInteractor().GetEventPosition()
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(click[0], click[1], 0)
            clickPos = coord.GetComputedWorldValue(renderer)
            #Update position
            self.click = [clickPos[0],clickPos[1],clickPos[2]]
            #Get camera focal point
            cam = renderer.GetActiveCamera()
            camFoc = cam.GetFocalPoint()
            camFocCoord = vtk.vtkCoordinate()
            camFocCoord.SetCoordinateSystemToDisplay()
            camFocCoord.SetValue(camFoc[0], camFoc[1], camFoc[2])
            foc = camFocCoord.GetComputedWorldValue(renderer)
            #Get camera position
            camPos = cam.GetPosition()
            camPosCoord = vtk.vtkCoordinate()
            camPosCoord.SetCoordinateSystemToDisplay()
            camPosCoord.SetValue(camPos[0], camPos[1], camPos[2])
            pos = camPosCoord.GetComputedWorldValue(renderer)
            #Update variables
            self.click = [clickPos[0],clickPos[1],clickPos[2]]
            self.pos = [pos[0],pos[1],pos[2]]
            self.foc = [foc[0],foc[1],-1]
            self.rightDown = True

        return
    
    def rightUpEvt(self,obj,event):
        if self.rightDown == True:
            self.rightDown = False
        return
    
    def Move2d(self,obj,event):
        self.cursor.__move__()
        if self.leftDown == True:
            #Get render window  components
            renWin = self.GetInteractor().GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            #Remove old contour actor
            renderer.RemoveActor(self.actor)
            #Compute click coordinates
            move = self.GetInteractor().GetEventPosition()           
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(move[0], move[1], 0)
            movePos = coord.GetComputedWorldValue(renderer)
            
            #Record point
            self.points[1,:] = np.array([movePos[0],movePos[1]])
            
            self.actor = make_actor(self.points)
            
            self.actor.GetProperty().SetLineWidth(self.lw)
            self.actor.GetProperty().SetColor(self.lc)
            #Update actor position
            actPos = self.actor.GetPosition()
            self.actor.SetPosition(actPos[0], actPos[1], -1)
            
            renderer.AddActor(self.actor)
            renWin.Render()
            
            jsw = (self.points[0,:]-self.points[1,:])**2
            jsw = (jsw.sum()**0.5)*0.148
            self.jsw = jsw
            
            
        if self.rightDown == True:
            #Get event position
            renWin = self.GetInteractor().GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            move = self.GetInteractor().GetEventPosition()
            #Compute position in world coordinates
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(move[0], move[1], -1)
            pos = coord.GetComputedWorldValue(renderer)
            #Compute new location, opposite of the movement direction
            self.pos = [self.pos[0]-(pos[0]-self.click[0]),
                        self.pos[1]-(pos[1]-self.click[1]),
                        self.pos[2]]
            #Update camera
            cam = renderer.GetActiveCamera()
            cam.SetPosition(self.pos[0],self.pos[1],self.pos[2])
            cam.SetFocalPoint(self.pos[0],self.pos[1],self.foc[2])
            renWin.Render()
            
        return
    
    def wheelFrwdEvt(self,obj,event):
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        cam = renderer.GetActiveCamera()
        cam.Zoom(1.1)
        renWin.Render()
        return
    
    def wheelBkwdEvt(self,obj,event):
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        cam = renderer.GetActiveCamera()
        cam.Zoom(0.9)
        renWin.Render()
        return
    
    def clear_actor(self):
        #Remove actor from renderwindow
        self.cursor.remove_cursor()
        renWin = self.GetInteractor().GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.actor)
        renderer.Modified()
        #Reset points/lines
        self.initialize()
        renWin.Render()
        
    def get_contour_components(self):
        actor_copy = vtk.vtkActor()
        actor_copy.ShallowCopy(self.actor)
        
        return actor_copy
'''