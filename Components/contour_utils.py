import numpy as np
import os
from copy import deepcopy
import vtk
from vtk.util import numpy_support


class knee_contours(object):
    def __init__(self,vtkWidget,lineWidth=4):
        #vtk polydata components
        
        #Contour actors
        self.tibiaLatCon = None
        self.tibiaMedCon = None
        self.femurLatCon = None
        self.femurMedCon = None
        
        self.vtkWidget = vtkWidget
        self.colors = [[0.0,1.0,0.0],[0.5,0.8,0.0],[0.0,0.0,1.0],[0.5,0.0,0.8]]
        self.lw = lineWidth
        
    def render_contours(self,choice='all'):
        choices = ['all','Tibia, Lateral','Tibia, Medial', 'Femur, Lateral', 'Femur, Medial']
        if choice not in choices:
            raise ValueError('Invalid controur specified, select one from %s' % choices)
        else:
            contours = [self.tibiaLatCon,self.tibiaMedCon,self.femurLatCon,self.femurMedCon]
            renWin = self.vtkWidget.GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            if choice == choices[0]:
                for contour,color in zip(contours,self.colors):
                    if renderer is not None and contour is not None:
                        contour.GetProperty().SetLineWidth(self.lw)
                        contour.GetProperty().SetColor(color)
                        renderer.AddActor(contour)
            elif choice == choices[1]:
                contour,color = contours[0],self.colors[0]
                if renderer is not None and contour is not None:
                    contour.GetProperty().SetLineWidth(self.lw)
                    contour.GetProperty().SetColor(color)
                    renderer.AddActor(contour)
            elif choice == choices[2]:
                contour,color = contours[1],self.colors[1]
                if renderer is not None and contour is not None:
                    contour.GetProperty().SetLineWidth(self.lw)
                    contour.GetProperty().SetColor(color)
                    renderer.AddActor(contour)
            elif choice == choices[3]:
                contour,color = contours[2],self.colors[2]
                if renderer is not None and contour is not None:
                    contour.GetProperty().SetLineWidth(self.lw)
                    contour.GetProperty().SetColor(color)
                    renderer.AddActor(contour)
            elif choice == choices[4]:
                contour,color = contours[3],self.colors[3]
                if renderer is not None and contour is not None:
                    contour.GetProperty().SetLineWidth(self.lw)
                    contour.GetProperty().SetColor(color)
                    renderer.AddActor(contour)
                
            renderer.Modified()
            renWin.Render()
        
        return
        
    def clear_contours(self,choice='all'):
        choices = ['all','Tibia, Lateral','Tibia, Medial', 'Femur, Lateral', 'Femur, Medial']
        if choice not in choices:
            raise ValueError('Invalid controur specified, select one from %s' % choices)
        else:
            renWin = self.vtkWidget.GetRenderWindow()
            renderer = renWin.GetRenderers().GetFirstRenderer()
            if renderer is not None:
                if choice == 'all':
                    actors = [self.tibiaLatCon,self.tibiaMedCon,self.femurLatCon,self.femurMedCon]
                    for actor in actors:
                        renderer.RemoveActor(actor)
                    self.tibiaLatCon = None
                    self.tibiaMedCon = None
                    self.femurLatCon = None
                    self.femurMedCon = None
                    
                elif choice == 'Tibia, Lateral':
                    renderer.RemoveActor(self.tibiaLatCon)
                    self.tibiaLatCon = None
                elif choice == 'Tibia, Medial':
                    renderer.RemoveActor(self.tibiaMedCon)
                    self.tibiaMedCon = None
                elif choice == 'Femur, Lateral':
                    renderer.RemoveActor(self.femurLatCon)
                    self.femurLatCon = None
                elif choice == 'Femur, Medial':
                    renderer.RemoveActor(self.femurMedCon)
                    self.femurMedCon = None
                renderer.Modified()
                renWin.Render()
            
        return
    
    def set_contour(self,polyDataActor,contour):
        choices = ['Tibia, Lateral','Tibia, Medial', 'Femur, Lateral', 'Femur, Medial']
        if contour not in choices:
            raise ValueError('Invalid contour specified, select one from %s' % choices)
        else:
            if contour == choices[0]:
                self.tibiaLatCon = None
                self.tibiaLatCon = polyDataActor
            elif contour == choices[1]:
                self.tibiaMedCon = None
                self.tibiaMedCon = polyDataActor
            elif contour == choices[2]:
                self.femurLatCon = None
                self.femurLatCon = polyDataActor
            elif contour == choices[3]:
                self.femurMedCon = None
                self.femurMedCon = polyDataActor
                
    def get_contours(self,choice):
        choices = ['all','Tibia, Lateral','Tibia, Medial', 'Femur, Lateral', 'Femur, Medial']
        if choice not in choices:
            raise ValueError('Invalid contour specified, select one from %s' % choices)
        if choice == 'all':
            actors = [self.tibiaLatCon,self.tibiaMedCon,self.femurLatCon,self.femurMedCon]
            names = ['Tibia, Lateral','Tibia, Medial', 'Femur, Lateral', 'Femur, Medial']
            return actors,names
        elif choice == 'Tibia, Lateral':
            return self.tibiaLatCon
        elif choice == 'Tibia, Medial':
            return self.tibiaMedCon, choice
        elif choice == 'Femur, Lateral':
            return self.femurLatCon, choice
        elif choice == 'Femur, Medial':
            return self.femurMedCon, choice
        
class landmark_points(object):
    def __init__(self,vtkWidget,colors=[[0.75,0.0,0.0],[0.0,0.75,0.0],[0.0,0.0,0.75],[0.0,0.75,0.75]],size=8):
        self.full = None; self.tibia = None; 
        self.femur = None; self.joint = None;
        self.vtkWidget = vtkWidget
        self.colors = colors
        self.size = size
        
    def render_points(self,pointActor,contour):
        self.actor=pointActor
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if contour == 'all':
            actors = [self.full,self.tibia,self.femur,self.joint]
            for actor,pactor,color in zip(actors,pointActor,self.colors):
                actor = pactor
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetPointSize(self.size)
                renderer.AddActor(actor)
        elif contour == 'full':
            self.full = pointActor
            self.full.GetProperty().SetColor(self.colors[0])
            self.full.GetProperty().SetPointSize(self.size)
            renderer.AddActor(self.full)
        elif contour == 'tibia':
            self.tibia = pointActor
            self.tibia.GetProperty().SetColor(self.colors[1])
            self.tibia.GetProperty().SetPointSize(self.size)
            renderer.AddActor(self.tibia)
        elif contour == 'femur':
            self.femur = pointActor
            self.femur.GetProperty().SetColor(self.colors[2])
            self.femur.GetProperty().SetPointSize(self.size)
            renderer.AddActor(self.femur)
        elif contour == 'joint':
            self.joint = pointActor
            self.joint.GetProperty().SetColor(self.colors[3])
            self.joint.GetProperty().SetPointSize(self.size)
            renderer.AddActor(self.joint)
        renderer.Modified()
        renWin.Render()
        
    def remove_points(self,contour):
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            if contour == 'all':
                actors = [self.full,self.tibia,self.femur,self.joint]
                for actor in actors:
                    renderer.RemoveActor(actor)
                    actor = None
            elif contour == 'full':
                renderer.RemoveActor(self.full)
                self.full = None
            elif contour == 'tibia':
                renderer.RemoveActor(self.tibia)
                self.tibia = None
            elif contour == 'femur':
                renderer.RemoveActor(self.femur)
                self.femur = None
            elif contour == 'joint':
                renderer.RemoveActor(self.joint)
                self.joint = None
        renderer.Modified()
        renWin.Render()
        
class tibia_rois(object):
    def __init__(self,vtkWidget,colors=[[1.0,0.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,0.0,1.0]],size=3):
        self.sbRight = None; self.tRight=None;
        self.sbLeft = None; self.tLeft=None;
        self.colors = colors;
        self.size=size
        self.vtkWidget = vtkWidget
        
    def set_actors(self,actors,names=['sbRight','tRight','sbLeft','tLeft']):
        for actor,name in zip(actors,names):
            if name=='sbRight':
                self.sbRight=actor
            if name=='tRight':
                self.tRight=actor
            if name=='sbLeft':
                self.sbLeft=actor
            if name=='tLeft':
                self.tLeft=actor
        
    def remove_contours(self,names=None):
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            if names is None:
                actors = [self.sbRight,self.tRight,self.sbLeft,self.tLeft]
                for actor in actors:
                    renderer.RemoveActor(actor)
                    actor = None
            else:
                for name in names:
                    if name=='sbRight':
                        renderer.RemoveActor(self.sbRight)
                        self.sbRight = None
                    if name=='tRight':
                        renderer.RemoveActor(self.tRight)
                        self.tRight = None
                    if name=='sbLeft':
                        renderer.RemoveActor(self.sbLeft)
                        self.sbLeft = None
                    if name=='tLeft':
                        renderer.RemoveActor(self.tLeft)
                        self.tLeft = None
        renderer.Modified()
        renWin.Render()
        
    def render_controus(self):
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            actors = [self.sbRight,self.tRight,self.sbLeft,self.tLeft]
            for actor,color in zip(actors,self.colors):
                if actor is not None:
                    actor.GetProperty().SetColor(color)
                    actor.GetProperty().SetLineWidth(self.size)
                    renderer.AddActor(actor)
        renderer.Modified()
        renWin.Render()
        
class edge_contours(object):
    def __init__(self,vtkWidget,colors=[[1.0,0.0,0.0],[0.0,0.0,1.0]],size=3):
        self.vtkWidget = vtkWidget
        self.colors = colors
        self.size = size
        self.tibia = None; self.femur=None
        
    def set_actors(self,actors,names=['tibia','femur']):
        for actor,name in zip(actors,names):
            if name == 'tibia':
                self.tibia = actor
            elif name == 'femur':
                self.femur = actor
    
    def remove_contours(self):
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            actors = [self.tibia,self.femur]
            for actor in actors:
                renderer.RemoveActor(actor)
                actor = None
        renderer.Modified()
        renWin.Render()
                
    def render_contours(self):
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            actors = [self.tibia,self.femur]
            for actor,color in zip(actors,self.colors):
                if actor is not None:
                    actor.GetProperty().SetColor(color)
                    actor.GetProperty().SetLineWidth(self.size)
                    renderer.AddActor(actor)
        renderer.Modified()
        renWin.Render()
        
class js_lines(object):
    def __init__(self,vtkWidget,colors=[[1.0,0.5,0.0],[0.0,5.0,1.0]],size=3):
        self.vtkWidget = vtkWidget
        self.colors = colors
        self.size = size
        self.lateral = None; self.medial=None
        
    def set_actors(self,actors,names=['lateral','medial']):
        for actor,name in zip(actors,names):
            if name == 'lateral':
                self.lateral = actor
            elif name == 'medial':
                self.medial = actor
    
    def remove_contours(self,names=None):
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            if names is None:
                actors = [self.lateral,self.medial]
                for actor in actors:
                    renderer.RemoveActor(actor)
                    actor = None
            else:
                for name in names:
                    if name == 'lateral':
                        renderer.RemoveActor(self.lateral)
                        self.lateral = None
                    if name == 'medial':
                        renderer.RemoveActor(self.medial)
                        self.medial = None
        renderer.Modified()
        renWin.Render()
                
    def render_contours(self):
        renWin = self.vtkWidget.GetRenderWindow()
        renderer = renWin.GetRenderers().GetFirstRenderer()
        if renderer is not None:
            actors = [self.lateral,self.medial]
            for actor,color in zip(actors,self.colors):
                if actor is not None:
                    actor.GetProperty().SetColor(color)
                    actor.GetProperty().SetLineWidth(self.size)
                    renderer.AddActor(actor)
        renderer.Modified()
        renWin.Render()