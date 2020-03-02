import numpy as np
import os
import vtk
from copy import deepcopy
from .processing import make_actor
from .vtk_utils import text_actor

class cursor(object):
    def __init__(self,interactorstyle,color=[1.0,0.0,0.0],R=5):
        self.interactorstyle = interactorstyle
        self.actor = vtk.vtkActor()
        self.lw = 3
        self.color = color
        self.R = R
    
    def __move__(self):
        #Get components
        interactor = self.interactorstyle.GetInteractor() 
        renWin = interactor.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old actor
        renderer.RemoveActor(self.actor)
        #Get mouse position
        pos = interactor.GetEventPosition()
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        coord.SetValue(pos[0], pos[1], 0)
        center = coord.GetComputedWorldValue(renderer)
        
        #Create line  objects
        points = vtk.vtkPoints()        
        lines = vtk.vtkCellArray()
        polydata = vtk.vtkPolyData()
        
        #Line positions
        x1,x2 = center[0]-self.R,center[0]+self.R
        y1,y2 = center[1]-self.R,center[1]+self.R
        
        id1 = points.InsertNextPoint([x1,y1,0])
        id2 = points.InsertNextPoint([x2,y2,0])
        
        lines.InsertNextCell(2)
        lines.InsertCellPoint(id1)
        lines.InsertCellPoint(id2)
        
        x1,x2 = center[0]-self.R,center[0]+self.R
        y1,y2 = center[1]+self.R,center[1]-self.R
        
        id1 = points.InsertNextPoint([x1,y1,0])
        id2 = points.InsertNextPoint([x2,y2,0])
        
        lines.InsertNextCell(2)
        lines.InsertCellPoint(id1)
        lines.InsertCellPoint(id2)
        
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        #Add to render window
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetLineWidth(self.lw)
        self.actor.GetProperty().SetColor(self.color)

        renderer.AddActor(self.actor)
        renWin.Render()
    
    def remove_cursor(self):
        #Get components
        interactor = self.interactorstyle.GetInteractor() 
        renWin = interactor.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        #Remove old actor
        renderer.RemoveActor(self.actor)
        renderer.Modified()

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
            renWin = self.GetInteractor().GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            #Get click location and compute world coordinates
            click = self.GetInteractor().GetEventPosition()
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(click[0], click[1], -1)
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
            self.foc = [foc[0],foc[1],0]
            self.leftDown = True

        return
    
    def leftUpEvt(self,obj,event):
        if self.leftDown == True:
            self.leftDown = False
        return
    
    def Move2d(self,obj,event):        
        if self.leftDown == True:
            #Get event position
            renWin = self.GetInteractor().GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            move = self.GetInteractor().GetEventPosition()
            #Compute position in world coordinates
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(move[0], move[1], 0)
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

class draw_interactor(vtk.vtkInteractorStyle):
    def __init__(self,line_width=3,line_color=[1.0,0.0,0.0]):
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
        
        self.initialize()
        
        self.cursor = cursor(self)
        
        
    def initialize(self):
        self.click = [0,0,0]
        self.pos = [0,0,0]
        self.foc = [0,0,0]
        
                #Contour componenets
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        #Rendering componenets
        self.contour = vtk.vtkPolyData()
        self.actor = vtk.vtkActor()
        
      
    def leftDownEvt(self,obj,event):
        if self.leftDown == False:
            #Get render window  components
            renWin = self.GetInteractor().GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            #Remove old contour actor
            renderer.RemoveActor(self.actor)
            #Compute click coordinates
            click = self.GetInteractor().GetEventPosition()            
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(click[0], click[1], 0)
            clickPos = coord.GetComputedWorldValue(renderer)
            #Add points/lines to contour
            id1 = self.points.InsertNextPoint([clickPos[0], clickPos[1], 0])
            id2 = self.points.InsertNextPoint([clickPos[0], clickPos[1], 0])
            self.lines.InsertNextCell(2)
            self.lines.InsertCellPoint(id1)
            self.lines.InsertCellPoint(id2)
            self.contour.SetPoints(self.points)
            self.contour.SetLines(self.lines)
            self.contour.Modified()
            
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.contour)
            
            #Add to render window
            self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetLineWidth(self.lw)
            self.actor.GetProperty().SetColor(self.lc)
            self.actor.Modified()
            renderer.AddActor(self.actor)
            renWin.Render()
            
            self.leftDown = True
            
        return
    
    def leftUpEvt(self,obj,event):
        if self.leftDown == True:
            self.leftDown = False
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
            #Compute number of current points
            Npoints = self.points.GetNumberOfPoints()
            #Add points/lines to contour
            id1 = self.points.InsertNextPoint(self.points.GetPoint(Npoints-1))
            id2 = self.points.InsertNextPoint([movePos[0], movePos[1], 0])
            self.lines.InsertNextCell(2)
            self.lines.InsertCellPoint(id1)
            self.lines.InsertCellPoint(id2)
            self.contour.SetPoints(self.points)
            self.contour.SetLines(self.lines)
            self.contour.Modified()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.contour)
            
            #Add to render window
            self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetLineWidth(self.lw)
            self.actor.GetProperty().SetColor(self.lc)
            #Update actor position
            actPos = self.actor.GetPosition()
            self.actor.SetPosition(actPos[0], actPos[1], -1)
            self.actor.GetProperty().SetLineWidth(3)
            self.actor.Modified()
            renderer.AddActor(self.actor)
            renWin.Render()
            
            
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