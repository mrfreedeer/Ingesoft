import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QInputDialog, QLineEdit, QDialog, QWidget, QGridLayout,QMessageBox, QLabel, QPushButton,QLineEdit, QSpinBox,QComboBox,QHBoxLayout
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap
from detecto import core, utils, visualize
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [1,10]


class Singleton(object):
    instances = {}
    def __new__(cls, clz = None):
        if clz is None:
            # print ("Creating object for", cls)
            if not cls.__name__ in Singleton.instances:
                Singleton.instances[cls.__name__] = \
                    object.__new__(cls)
            return Singleton.instances[cls.__name__]
        # print (cls.__name__, "creating", clz.__name__)
        Singleton.instances[clz.__name__] = clz()
        Singleton.first = clz
        return type(clz.__name__, (Singleton,), dict(clz.__dict__))



class main_window(QtWidgets.QMainWindow):
    __metaclass__= Singleton()
    def __init__(self):
        #Iniciar objeto
        super(main_window, self).__init__()
        #Cargar GUI
        uic.loadUi("main.ui", self)
        #Cargar la configuracion del archivo .ui en el objeto
        self.title = 'Aguacate'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.loadedimages = None
        self.model = core.Model.load('avocado_weights.pth', ['fit_avocado', 'unfit_avocado'])
        #Usar botón Cargar imagen
        self.button_load = self.findChild(QtWidgets.QPushButton, 'load_button')
        self.button_load.clicked.connect(self.openFileNamesDialog)
        self.button_load_many = self.findChild(QtWidgets.QPushButton, 'load_many_button')
        self.button_load_many.clicked.connect(self.loadFolder)
        #self.initUI()

        #Usar botón de ejecutar
        self.button_exec = self.findChild(QtWidgets.QPushButton, 'exec_button')
        self.button_exec.clicked.connect(self.predict)
        

        #Abrir archivo y mostrar imagen
    def openFileNamesDialog(self):

        #Abrir archivo
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)
        self.imagePath = files[0]
        print (self.imagePath)
        pixmap = QPixmap(self.imagePath)
        #Escalar archivo
        pixmap2 = pixmap.scaled(190, 140)
        #Imprimir imagen en pantalla
        self.show_image.setPixmap(pixmap2)
    def loadFolder(self):
        #Abrir archivo
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        #files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        #if files:
        #    print(files)
        #imagePath = files[0]
        

        image_list = []
        for filename in glob.glob('images/*.jpg'): #asumiendo jpg
            im=Image.open(filename)
            image_list.append(im)
        self.loadedimages = image_list
        QMessageBox.information(self, "Cargando Carpeta",
                                        "Se han cargado todas las imagenes")
        # for element in image_list:
        #     print (element)
        #pixmap = QPixmap(imagePath)
        #Escalar archivo
        #pixmap2 = pixmap.scaled(190, 140)
        #Imprimir imagen en pantalla
        #self.show_image.setPixmap(pixmap2)

        

            
    def closeEvent(self,event):
      reply =QMessageBox.question(self, "Mensaje", "Seguro quiere salir", QMessageBox.Yes, QMessageBox.No)
   
      if reply == QMessageBox.Yes:
       event.accept()
      else: 
        event.ignore()


    def predict(self):
        if self.loadedimages:
            self.predict_many()
        else:
            self.predict_one()
    def predict_one (self):
      
      
        # Specify the path to your image
        print (self.imagePath)
        image = utils.read_image(self.imagePath)
        predictions = self.model.predict(image)
        
        # predictions format: (labels, boxes, scores)
        labels, boxes, scores = predictions
        
        # ['alien', 'bat', 'bat']
        print(labels) 
        
        #           xmin       ymin       xmax       ymax
        # tensor([[ 569.2125,  203.6702, 1003.4383,  658.1044],
        #         [ 276.2478,  144.0074,  579.6044,  508.7444],
        #         [ 277.2929,  162.6719,  627.9399,  511.9841]])
        print(boxes)
        
        # tensor([0.9952, 0.9837, 0.5153])
        print(scores)
        visualize.show_labeled_image(image, boxes, labels)
    
    def predict_many(self):
        fit_avocados = 0
        unfit_avocados = 0
        QMessageBox.information(self, "Listo para empezar",
                                        "Calculando productividad")
        for image in self.loadedimages:
            labels, boxes, scores = self.model.predict(image)
            fit_avocados += labels.count('fit_avocado')
            unfit_avocados += labels.count('unfit_avocado')
        total = fit_avocados+unfit_avocados
        print("Total: ", total)
        print("Productivity: ", fit_avocados/total)
        
        
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Aguacates maduros y listos', 'Otros Aguacates'
        sizes = [fit_avocados/total, unfit_avocados/total]
        explode = (0.1, 0)  

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        fig1.canvas.set_window_title('Productividad de la cosecha')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = main_window()
    main.show()

    main.setWindowTitle('Bienvenido')
    app.exec_()
    
        
        
