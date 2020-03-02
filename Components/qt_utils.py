import numpy as np
import os
from copy import deepcopy
import time
import vtk
import multiprocessing
import threading
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow,
                             QVBoxLayout, QFrame, QFileDialog, QMenu,
                             QMenuBar, QAction, QSizePolicy, qApp)

from .vtk_utils import xray_rendering,make_polyDataActor
from .IO_utils import *
from .contour_utils import *
from pygments.lexers._vim_builtins import option
from .measurement_utils import kneeAnalyzer
from .thread_utils import analyze_all,analyze_current,train_model,load_model
from Components.processing import get_line_ends, get_points

class guitext(object):
    def __init__(self,separator = ":",spaces=4):
        self.line_titles = []
        self.line_content = []
        self.sep = separator
        self.spaces = spaces
        for k in range(spaces):
            self.sep += " "
    
    def __call__(self):
        #Get longest title
        N = 0
        for title in self.line_titles:
            if len(title)>N:
                N = len(title)
        txt = ""
        for title,content in zip(self.line_titles,self.line_content):
            delta = N-len(title)
            for k in range(delta+self.spaces):
                title += " "
            txt += title+self.sep+content+"\n"
        return txt
    
    def add_line(self,title,content=""):
        self.line_titles.append(title)
        self.line_content.append(content)
        
    def set_content(self,title,content):
        if title not in self.line_titles:
            raise ValueError("Title not in current!! Titles: %s" % self.line_titles)
        #Update content
        for k in range(len(self.line_titles)):
            title_ = self.line_titles[k]
            if title == title_:
                self.line_content[k] = content
        

class main_window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.w = 800
        self.h = 800
        self.left = 100
        self.top = 100
        self.title = 'KneeAnalyzer'
        
        self.cur_idx = 0
        self.cur_side = 'R'
        self.cur_model = 'full'
        self.cur_drawname = 'lateral_js'
        self.has_loaded = 'No Contours!!'
        
        self.annotations = './annotations'
        self.models = './models'
        self.results = './results'
        self.manual_results = './manual'
        
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        #Create rendering object
        self.xray = xray_rendering(self.vtkWidget,functions_=[self.get_line,self.get_box])
        
        #Create dicom object
        self.dicom = DICOM()
        #Create model contour objects
        self.contours = knee_contours(self.vtkWidget)
        self.landmarks = landmark_points(self.vtkWidget)
        self.segmentation = landmark_points(self.vtkWidget)
        self.points = landmark_points(self.vtkWidget)
        self.rois_automatic = tibia_rois(self.vtkWidget,colors=[[0.0,0.5,0.5],[0.0,1.0,1.0],
                                                                [0.0,0.5,0.5],[0.0,1.0,1.0]])
        self.rois_manual = tibia_rois(self.vtkWidget,colors=[[0.0,0.0,0.5],[0.0,0.0,1.0],
                                                             [0.0,0.0,0.5],[0.0,0.0,1.0]])
        self.lateral_edges = edge_contours(self.vtkWidget)
        self.medial_edges = edge_contours(self.vtkWidget)
        self.jsw_lines_automatic = js_lines(self.vtkWidget,colors=[[0.0,1.0,1.0],[0.0,1.0,1.0]])
        self.jsw_lines_manual = js_lines(self.vtkWidget,colors=[[0.0,0.0,1.0],[0.0,0.0,1.0]])
        #Create analyzer
        self.analyzer = kneeAnalyzer()
        
        #Create GUI text
        self.txt = guitext()
        self.guititles = ['Subject','Contours','Points','Landmarks','ROIs','Segmentation','Lateral JSW','Medial JSW','Analysis']
        self.guicontent = ['Not Loaded','Not Loaded','Not Loaded','Not Loaded','Not Loaded','Not Loaded','-','-','Not running']
        for title,content in zip(self.guititles,self.guicontent):
            self.txt.add_line(title, content)
        
        self.initGUI()
        
    def initGUI(self):
        #Helper variables
        self.cur_iactor = 0
        #Set window geometry and title        
        self.setGeometry(self.left,self.top,self.w,self.h)
        self.setWindowTitle(self.title)
        
        #Add menus
        menuBar = self.menuBar()
        
        #File menu
        fileMenu = QMenu('File',self)
        
        #DICOM loader
        loadFileAction = QAction('Load DICOM(s)',self)
        loadFileAction.triggered.connect(self.loadDicomDialog)
        loadFileAction.setShortcut('Ctrl+f')
        loadFileAction.setToolTip('Load dicom file(s) (Ctrl+f)')
        fileMenu.addAction(loadFileAction)
        #Annotation saving
        saveContourAction = QAction('Save contours',self)
        saveContourAction.triggered.connect(self.save_annotations)
        saveContourAction.setShortcut('Ctrl+s')
        saveContourAction.setToolTip('Save annotations (Ctrl+s)')
        fileMenu.addAction(saveContourAction)
        #Annotation loading
        loadContourAction = QAction('Load contours',self)
        loadContourAction.triggered.connect(self.load_annotation)
        loadContourAction.setShortcut('Ctrl+a')
        loadContourAction.setToolTip('Open existing annotations (Ctrl+a)')
        fileMenu.addAction(loadContourAction)
        #Load points from a file
        loadPointsAction = QAction('Load points from file',self)
        loadPointsAction.triggered.connect(self.load_pointsDialog)
        fileMenu.addAction(loadPointsAction)
        #Exit program
        exitAction = QAction('Exit',self)
        exitAction.triggered.connect(qApp.exit)
        exitAction.setShortcut('Ctrl+q')
        exitAction.setToolTip('Exit application (Ctrl+q)')
        fileMenu.addAction(exitAction)
        
        #Draw menu
        drawMenu = QMenu('Draw',self)
        
        #Draw annotations
        drawAction = QAction('Switch interactor',self)
        drawAction.setShortcut('Ctrl+d')
        drawAction.triggered.connect(self.switch_interactor)
        drawMenu.addAction(drawAction)
        drawTibiaLat = QAction('Get Tibia, Lateral',self)
        drawTibiaLat.setShortcut('Ctrl+1')
        drawTibiaLat.triggered.connect(self.get_tibia_lat)
        drawMenu.addAction(drawTibiaLat)
        drawTibiaMed = QAction('Get Tibia, Medial',self)
        drawTibiaMed.setShortcut('Ctrl+2')
        drawTibiaMed.triggered.connect(self.get_tibia_med)
        drawMenu.addAction(drawTibiaMed)
        drawFemurLat = QAction('Get Femur, Lateral',self)
        drawFemurLat.setShortcut('Ctrl+3')
        drawFemurLat.triggered.connect(self.get_femur_lat)
        drawMenu.addAction(drawFemurLat)
        drawFemurMed = QAction('Get Femur, Medial',self)
        drawFemurMed.setShortcut('Ctrl+4')
        drawFemurMed.triggered.connect(self.get_femur_med)
        drawMenu.addAction(drawFemurMed)
        #Clear annotations
        clearDrawnAction = QAction('Clear annotations',self)
        clearDrawnAction.setShortcut('Ctrl+C')
        clearDrawnAction.triggered.connect(self.clear_annotation)
        drawMenu.addAction(clearDrawnAction)
        
        #Display Menu
        dispMenu = QMenu('Show',self)
        #Select side
        dispLeftAction = QAction('Show left knee',self)
        dispLeftAction.setShortcut('Ctrl+t')
        dispLeftAction.triggered.connect(self.switch_image_left)
        dispMenu.addAction(dispLeftAction)
        dispRightAction = QAction('Show right knee',self)
        dispRightAction.setShortcut('Ctrl+r')
        dispRightAction.triggered.connect(self.switch_image_right)
        dispMenu.addAction(dispRightAction)
        #Select next/previous file
        dispNextAction = QAction('Show next knee',self)
        dispNextAction.setShortcut('Ctrl+n')
        dispNextAction.triggered.connect(self.switch_image_next)
        dispMenu.addAction(dispNextAction)
        dispPrevAction = QAction('Show previous knee',self)
        dispPrevAction.setShortcut('Ctrl+b')
        dispPrevAction.triggered.connect(self.switch_image_previous)
        dispMenu.addAction(dispPrevAction)
        
        #Analysis menu
        analysisMenu = QMenu('Analysis',self)
        #Train ASM
        trainASMAction = QAction('Train ASM',self)
        trainASMAction.triggered.connect(self.trainModels)
        analysisMenu.addAction(trainASMAction)
        #Load existing
        loadASMAction = QAction('Load ASM',self)
        loadASMAction.triggered.connect(self.loadModels)
        analysisMenu.addAction(loadASMAction)
        #Analyze files
        measureCurrentAction = QAction('Measure Current',self)
        measureCurrentAction.triggered.connect(self.analyzeCurrent)
        analysisMenu.addAction(measureCurrentAction)
        measureAllAction = QAction('Measure All',self)
        measureAllAction.triggered.connect(self.analyzeAll)
        analysisMenu.addAction(measureAllAction)
        
        #Add menus and actions for result visualization
        modelMenu = QMenu('Load landmarks',self)
        drawModelFull = QAction('Full knee',self)
        drawModelFull.triggered.connect(self.draw_model_full)
        modelMenu.addAction(drawModelFull)
        drawModelTibia = QAction('Tibia',self)
        drawModelTibia.triggered.connect(self.draw_model_tibia)
        modelMenu.addAction(drawModelTibia)
        drawModelFemur = QAction('Femur',self)
        drawModelFemur.triggered.connect(self.draw_model_femur)
        modelMenu.addAction(drawModelFemur)
        drawModelJoint = QAction('Joint',self)
        drawModelJoint.triggered.connect(self.draw_model_joint)
        modelMenu.addAction(drawModelJoint)
        
        resultMenu = QMenu('Load automatic segmentation',self)
        drawResultFull = QAction('Full knee',self)
        drawResultFull.triggered.connect(self.draw_result_full)
        resultMenu.addAction(drawResultFull)
        drawResultTibia = QAction('Tibia',self)
        drawResultTibia.triggered.connect(self.draw_result_tibia)
        resultMenu.addAction(drawResultTibia)
        drawResultFemur = QAction('Femur',self)
        drawResultFemur.triggered.connect(self.draw_result_femur)
        resultMenu.addAction(drawResultFemur)
        drawResultJoint = QAction('Joint',self)
        drawResultJoint.triggered.connect(self.draw_result_joint)
        resultMenu.addAction(drawResultJoint)
        #Show ROIs
        roiAutomatic = QAction('Show automatic ROI',self)
        roiAutomatic.triggered.connect(self.load_roi_automatic)
        #Show JSW masks
        readJSMask = QAction('Show joint masks',self)
        readJSMask.triggered.connect(self.load_jsw_mask)
        #Show edges
        readEdges = QAction('Show edges',self)
        readEdges.triggered.connect(self.load_jsedges)
        #Show JSW lines
        readLines = QAction('Show JS lines',self)
        readLines.triggered.connect(self.load_jslines_automatic)
        
        analysisMenu.addMenu(modelMenu)
        analysisMenu.addMenu(resultMenu)
        analysisMenu.addAction(readLines)
        analysisMenu.addAction(readEdges)
        analysisMenu.addAction(roiAutomatic)
        analysisMenu.addAction(readJSMask)
        
        #Add manual analysis menu
        manualMenu = QMenu('Manual measurement',self)
        lateralWidth = QAction('Measure lateral JSW',self)
        lateralWidth.triggered.connect(self.switch_interactor_line_lat)
        manualMenu.addAction(lateralWidth)
        medialWidth = QAction('Measure medial JSW',self)
        medialWidth.triggered.connect(self.switch_interactor_line_med)
        manualMenu.addAction(medialWidth)
        lateralROIsmall = QAction('Set lateral ROI (small)',self)
        lateralROIsmall.triggered.connect(self.switch_interactor_box_small_lat)
        manualMenu.addAction(lateralROIsmall)
        lateralROIbig = QAction('Set lateral ROI (large)',self)
        lateralROIbig.triggered.connect(self.switch_interactor_box_big_lat)
        manualMenu.addAction(lateralROIbig)
        medialROIsmall = QAction('Set medial ROI (small)',self)
        medialROIsmall.triggered.connect(self.switch_interactor_box_small_med)
        manualMenu.addAction(medialROIsmall)
        medialROIbig = QAction('Set medial ROI (large)',self)
        medialROIbig.triggered.connect(self.switch_interactor_box_big_med)
        manualMenu.addAction(medialROIbig)
        saveManual = QAction('Save analyses',self)
        saveManual.triggered.connect(self.save_manual_results)
        manualMenu.addAction(saveManual)
        loadManual = QAction('Load analyses',self)
        loadManual.triggered.connect(self.load_manual_results)
        manualMenu.addAction(loadManual)
        
        #Add menus to the program
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(drawMenu)
        menuBar.addMenu(dispMenu)
        menuBar.addMenu(analysisMenu)
        menuBar.addMenu(manualMenu)
        #Create layout, frame and widget for vtk rendering
        self.vtkLayout = QVBoxLayout()
        self.vtkFrame = QFrame()
        #Add vtkWidget to the frame and set layout
        self.vtkLayout.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vtkLayout)
        #Set central widget for the application
        self.setCentralWidget(self.vtkFrame)
        #Show application and start render window interactor
        self.show()
        self.iren.Start()
        
    def trainModels(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file,path = QFileDialog.getOpenFileName(self, 'Open grade file', '',
                                                  'csv(*.csv *.xls *xlsx)', options=options)
        if file:
            title = 'Analysis'; content = 'Running..'
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
            t = threading.Thread(target=train_model, args=(self,file))
            t.daemon=True;t.start();
        
    def loadModels(self):
        title = 'Analysis'; content = 'Running..'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
        t = threading.Thread(target=load_model, args=(self,))
        t.daemon=True;t.start();
        
        
    def analyzeAll(self):
        if self.analyzer.model is None:
            title = 'Analysis'; content = 'Train or load ASM first!!'
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
        else:
            title = 'Analysis'; content = 'Running..'
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
            t = threading.Thread(target=analyze_all, args=(self,))
            t.daemon=True;t.start();
    
    def analyzeCurrent(self):
        if self.analyzer.model is None:
            title = 'Analysis'; content = 'Train or load ASM first!!'
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
        else:
            title = 'Analysis'; content = 'Running..'
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
            image = self.dicom.pixels(self.cur_side)
            name = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side
            t = threading.Thread(target=analyze_current, args=(self,image,name))
            t.daemon=True;t.start();
        
        
    def loadDicomDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files,path = QFileDialog.getOpenFileNames(self, 'Open Files(s)', '',
                                                  'All files(*)', options=options)
        if files:
            self.dicom.files,self.dicom.path = files,path
            self.dicom.read_file(self.cur_idx)
            self.xray.clear_actors()
            self.xray.render_image(self.dicom.pixels(self.cur_side))
            #Update UI Text
            for title,content in zip(self.guititles,self.guicontent):
                self.txt.set_content(title, content)
            title = 'Subject'; content = self.dicom.get_name(self.cur_idx)+'    |    '+str(self.cur_side)
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
        
    def save_annotations(self):
        actors,names = self.contours.get_contours('all')
        filename = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side
        contours_to_csv(actors, names, filename)
        #Update UI Text
        title = 'Contours'; content = 'Saved!'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
        
    def load_annotation(self):
        filename = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side+'.csv'
        annotation_files = os.listdir('./annotations')
        if filename in annotation_files:
            self.contours.clear_contours('all')
            contours, names = csv_to_contour(filename)
            for contour,name in zip(contours,names):
                self.contours.set_contour(contour, name)
                self.contours.render_contours(name)
            self.xray.clear_draw_actor()
            #Update UI Text
            title = 'Contours'; content = 'Loaded!'
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
            
    def save_manual_results(self):
        name = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side
        lateral = self.jsw_lines_manual.lateral
        medial = self.jsw_lines_manual.medial
        sbRight = self.rois_manual.sbRight
        tRight = self.rois_manual.tRight
        sbLeft = self.rois_manual.sbLeft
        tLeft = self.rois_manual.tLeft
        
        save_manual_results(name, self.manual_results, lateral, medial, sbRight, tRight, sbLeft, tLeft)
        
        return
    
    def load_manual_results(self):
        name = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side
        pointname = name+'_points.csv'
        if os.path.isfile(os.path.join(self.manual_results,'JS_lines',pointname)):
            lateral,medial,Dlat,Dmed = read_js_lines(os.path.join(self.manual_results,'JS_lines',name))
            self.jsw_lines_manual.remove_contours()
            self.jsw_lines_manual.set_actors([lateral,medial], ['lateral','medial'])
            self.jsw_lines_manual.render_contours()
            if Dlat is not None:
                title = 'Lateral JSW'; content = '(manual) {:.3f} (mm)'.format(Dlat)
                self.txt.set_content(title, content)
            if Dmed is not None:
                title = 'Medial JSW'; content = '(manual) {:.3f} (mm)'.format(Dmed)
                self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
        
        roiname = os.path.join(self.manual_results,'ROIs',name+'.csv')
        if os.path.isfile(roiname):
            sbRigth,tRight,sbLeft,tLeft = csv_to_roi(roiname)
            self.rois_manual.remove_contours()
            self.rois_manual.set_actors([sbRigth,tRight,sbLeft,tLeft], ['sbRight','tRight','sbLeft','tLeft'])
            self.rois_manual.render_controus()

    def load_pointsDialog(self):
        #titles = ['Subject','Contours','Points','Model','Segmentation','Lateral JSW','Medial JSW']
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file,path = QFileDialog.getOpenFileName(self, 'Open csv', '',
                                                '.csv(*.csv)', options=options)
        
        if file:
            self.points.remove_points('full')
            actor = csv_to_point_actor(file)
            self.points.render_points(actor,'full')
            title = 'Points'; content = file
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
            
    def clear_annotation(self):
        self.jsw_lines_automatic.remove_contours()
        self.jsw_lines_manual.remove_contours()
        self.lateral_edges.remove_contours()
        self.medial_edges.remove_contours()
        self.rois_automatic.remove_contours()
        self.rois_manual.remove_contours()
        self.points.remove_points('all')
        self.landmarks.remove_points('all')
        self.segmentation.remove_points('all')
        self.contours.clear_contours('all')
        self.xray.clear_draw_actor()
        self.xray.clearmask()
        for title,content in zip(self.guititles[1:-1],self.guicontent[1:-1]):
            self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
            
    def switch_interactor_image(self):
        self.xray.set_interactor('image')
        self.cur_iactor = 0
        
    def switch_interactor_draw(self):
        self.xray.set_interactor('draw')
        self.cur_iactor = 1
        
    def switch_interactor_line_lat(self):
        self.xray.set_interactor('line')
        self.cur_drawname = 'lateral_js'
        self.cur_iactor = 1
        
    def switch_interactor_line_med(self):
        self.xray.set_interactor('line')
        self.cur_drawname = 'medial_js'
        self.cur_iactor = 1
        
    def switch_interactor_box_small_lat(self):
        self.xray.set_interactor('box_small')
        self.cur_drawname = 'sbRight'
        self.cur_iactor = 1
        
    def switch_interactor_box_big_lat(self):
        self.xray.set_interactor('box_big')
        self.cur_drawname = 'tRight'
        self.cur_iactor = 1
        
    def switch_interactor_box_small_med(self):
        self.xray.set_interactor('box_small')
        self.cur_drawname = 'sbLeft'
        self.cur_iactor = 1
        
    def switch_interactor_box_big_med(self):
        self.xray.set_interactor('box_big')
        self.cur_drawname = 'tLeft'
        self.cur_iactor = 1
        
    def switch_interactor(self):
        if self.cur_iactor == 0:
            self.switch_interactor_draw()
        elif self.cur_iactor == 1:
            self.xray.clear_draw_actor()
            self.switch_interactor_image()
        
    def get_tibia_lat(self):
        actor = self.xray.get_draw_actor()
        self.contours.clear_contours('Tibia, Lateral')
        self.contours.set_contour(actor, 'Tibia, Lateral')
        self.contours.render_contours('Tibia, Lateral')
        self.xray.clear_draw_actor()
        title = 'Contours'; content = 'Drawn'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())  
        
        return
    
    def get_tibia_med(self):
        actor = self.xray.get_draw_actor()
        self.contours.clear_contours('Tibia, Medial')
        self.contours.set_contour(actor, 'Tibia, Medial')
        self.contours.render_contours('Tibia, Medial')
        self.xray.clear_draw_actor()
        title = 'Contours'; content = 'Drawn'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
              
        return
    
    def get_femur_lat(self):
        actor = self.xray.get_draw_actor()
        self.contours.clear_contours('Femur, Lateral')
        self.contours.set_contour(actor, 'Femur, Lateral')
        self.contours.render_contours('Femur, Lateral')
        self.xray.clear_draw_actor()
        title = 'Contours'; content = 'Drawn'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
           
        return
    
    def get_femur_med(self):
        actor = self.xray.get_draw_actor()
        self.contours.clear_contours('Femur, Medial')
        self.contours.set_contour(actor, 'Femur, Medial')
        self.contours.render_contours('Femur, Medial')
        self.xray.clear_draw_actor()
        title = 'Contours'; content = 'Drawn'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
         
        return
    
    def get_line(self):
        actor = self.xray.get_draw_actor()
        if self.cur_drawname == 'lateral_js':
            names = ['lateral']
            title = 'Lateral JSW'
        elif self.cur_drawname == 'medial_js':
            names = ['medial']
            title = 'Medial JSW'
        self.jsw_lines_manual.remove_contours(names)
        self.jsw_lines_manual.set_actors([actor], names)
        self.jsw_lines_manual.render_contours()
        self.switch_interactor()
        self.cur_iactor = 0
        
        points = get_line_ends(actor)
        D = (points[0,:]-points[1,:])**2
        D = D.sum()**0.5
        
        content = "(manual) {:.3f} mm".format(D*0.148)
        
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
        
        
        return
        
    def get_box(self):
        #Get draw actor
        actor = self.xray.get_draw_actor()
        if self.cur_drawname == 'sbRight':
            names = ['sbRight']
        if self.cur_drawname == 'tRight':
            names = ['tRight']
        if self.cur_drawname == 'sbLeft':
            names = ['sbLeft']
        if self.cur_drawname == 'tLeft':
            names = ['tLeft']
        #Clear render window
        self.rois_manual.remove_contours(names)
        self.rois_manual.set_actors([actor], names)
        self.rois_manual.render_controus()
        
        self.switch_interactor()
        self.cur_iactor = 0

        return

    
    def draw_model_full(self):
        self.cur_model = 'full'
        self.load_model('Points')
        
    def draw_model_tibia(self):
        self.cur_model = 'tibia'
        self.load_model('Points')
        
    def draw_model_femur(self):
        self.cur_model = 'femur'
        self.load_model('Points')
        
    def draw_model_joint(self):
        self.cur_model = 'joint'
        self.load_model('Points')
        
    def draw_result_full(self):
        self.cur_model = 'full'
        self.load_model('segmentation')
        
    def draw_result_tibia(self):
        self.cur_model = 'tibia'
        self.load_model('segmentation')
        
    def draw_result_femur(self):
        self.cur_model = 'femur'
        self.load_model('segmentation')
        
    def draw_result_joint(self):
        self.cur_model = 'joint'
        self.load_model('segmentation')
        
        
    def load_model(self,type_):
        if type_=='Points':
            sample = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side+'_'+self.cur_model+'.csv'
            filepath = os.path.join(self.models,type_,self.cur_model,sample)
            if os.path.isfile(filepath):
                actor = csv_to_point_actor(filepath,new_path=True)
                self.landmarks.remove_points(self.cur_model)
                self.landmarks.render_points(actor,self.cur_model)
                content = 'Loaded'
            else:
                content = 'Not found'               
            title = 'Landmarks'
        elif type_=='segmentation':
            sample = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side+'_'+self.cur_model+'.csv'
            filepath = os.path.join(self.results,type_,self.cur_model,sample)
            if os.path.isfile(filepath):
                actor = csv_to_point_actor(filepath,new_path=True)
                self.segmentation.remove_points(self.cur_model)
                self.segmentation.render_points(actor,self.cur_model)
                content = 'Loaded'
            else:
                content = 'Not found'
            title = 'Segmentation'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
        
    def load_roi_automatic(self):
        sample = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side+'.csv'
        filepath = os.path.join(self.results,'ROIs',sample)
        title = 'ROIs'
        if os.path.isfile(filepath):
            roiactors = csv_to_roi(filepath)
            self.rois_automatic.set_actors(roiactors)
            self.rois_automatic.render_controus()
            content = 'Loaded'
        else:
            content = 'Not found'
        self.txt.set_content(title, content)
        self.xray.render_text(self.txt())
        
    def load_jsw(self):
        sample = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side+'.csv'
        path = os.path.join(self.results,'JS_widths',sample)
        if os.path.isfile(path):
            lateral,medial = read_jsw(path)
            #'Lateral JSW','Medial JSW'
            title = 'Lateral JSW'; content = '(mean) {:.3f} mm'.format(lateral)
            self.txt.set_content(title, content)
            title = 'Medial JSW'; content = '(mean) {:.3f} mm'.format(medial)
            self.txt.set_content(title, content)
            self.xray.render_text(self.txt())
            
    def load_jslines_automatic(self):
        #self.guititles = ['Lateral JSW','Medial JSW']
        sample = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side
        path = os.path.join(self.results,'JS_edges',sample)
        if os.path.isfile(path+'_points.csv'):
            pl,pm,dl,dm = read_js_lines(path)
            self.jsw_lines_automatic.remove_contours()
            self.jsw_lines_automatic.set_actors([pl,pm])
            self.jsw_lines_automatic.render_contours()
            
            title = 'Lateral JSW'; content = '(automatic) {:.3f} mm'.format(dl);
            self.txt.set_content(title, content)
            
            title = 'Medial JSW'; content = '(automatic) {:.3f} mm'.format(dm);
            self.txt.set_content(title, content)
            
            self.xray.render_text(self.txt())
        return
            
    def load_jsedges(self):
        #self.guititles = ['Lateral JSW','Medial JSW']
        sample = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side
        path = os.path.join(self.results,'JS_edges',sample)
        if os.path.isfile(path+'_edges.csv'):
            lt,mt,lf,mf = read_js_edges(path)
            self.lateral_edges.set_actors([lt,lf])
            self.lateral_edges.render_contours()
            self.medial_edges.set_actors([mt,mf])
            self.medial_edges.render_contours()
        return
            
    def load_jsw_mask(self):
        sample = self.dicom.get_name(self.cur_idx)+'_'+self.cur_side
        path = os.path.join(self.results,'JSW_masks',sample)
        if os.path.isfile(path+'.csv'):
            mask,x,y = read_jsw_mask(path)
            self.xray.clearmask()
            self.xray.rendermask(mask, x, y)
            self.load_jsw()
            
    
    def clear_drawn(self):
        self.xray.clear_draw_actor()
        
    def clear_all(self):
        self.points.remove_points()
        self.landmarks.remove_points()
        self.segmentation.remove_points()
        self.xray.clear_draw_actor()
        self.contours.clear_contours('all')
        
        
    def switch_image_next(self):
        N = self.dicom.get_number_of_files()
        if self.cur_idx<(N-1):
            self.cur_idx += 1
            self.cur_side = 'R'
            self.dicom.read_file(self.cur_idx)
            self.xray.clear_actors()
            self.xray.render_image(self.dicom.pixels(self.cur_side))
            title = self.guititles[0]; content = self.dicom.get_name(self.cur_idx)+'    |    '+str(self.cur_side);
            self.txt.set_content(title, content)
            self.clear_annotation()
            #Rendering new image switches the interactor to the default
            self.cur_iactor = 0
            return
    
    def switch_image_previous(self):
        if self.cur_idx>0:
            self.cur_idx -= 1
            self.cur_side = 'R'
            self.dicom.read_file(self.cur_idx)
            self.xray.clear_actors()
            self.xray.render_image(self.dicom.pixels(self.cur_side))
            title = self.guititles[0]; content = self.dicom.get_name(self.cur_idx)+'    |    '+str(self.cur_side);
            self.txt.set_content(title, content)
            self.clear_annotation()
            #Rendering new image switches the interactor to the default
            self.cur_iactor = 0
            return
        
    def switch_image_left(self):
        #Update side
        if self.cur_side=='R':
            self.cur_side = 'L'
            #Load new image    
            self.dicom.read_file(self.cur_idx)
            self.xray.clear_actors()
            self.xray.render_image(self.dicom.pixels(self.cur_side))
            title = self.guititles[0]; content = self.dicom.get_name(self.cur_idx)+'    |    '+str(self.cur_side)+' MIRRORED';
            self.txt.set_content(title, content)
            self.clear_annotation()
            #Rendering new image switches the interactor to the default
            self.cur_iactor = 0
        return
            
    def switch_image_right(self):
        if self.cur_side == 'L':
            self.cur_side = 'R'
            #Load new image    
            self.dicom.read_file(self.cur_idx)
            self.xray.clear_actors()
            self.xray.render_image(self.dicom.pixels(self.cur_side))
            title = self.guititles[0]; content = self.dicom.get_name(self.cur_idx)+'    |    '+str(self.cur_side);
            self.txt.set_content(title, content)
            self.clear_annotation()
            #Rendering new image switches the interactor to the default
            self.cur_iactor = 0
        return