import time

def analyze_current(class_,image,name,title='Analysis',content='Done!'):
    class_.analyzer(image,name)
    class_.txt.set_content(title,content)
    class_.xray.render_text(class_.txt())
    
def analyze_all(class_,title='Analysis',content='Done!'):
    class_.analyzer.measure_all(class_.dicom,guiclass_=class_)
    class_.txt.set_content(title,content)
    class_.xray.render_text(class_.txt())
    
def train_model(class_,gradepath,title='Analysis',content='Done!'):
    class_.txt.set_content(title,gradepath)
    class_.xray.render_text(class_.txt())
    time.sleep(2)
    class_.analyzer.train_models(gradepath)
    class_.txt.set_content(title,content)
    class_.xray.render_text(class_.txt())
    
def load_model(class_,title='Analysis',content='ASM Loaded'):
    class_.analyzer.load_models()
    class_.txt.set_content(title,content)
    class_.xray.render_text(class_.txt())