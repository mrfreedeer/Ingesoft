import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QInputDialog, QLineEdit, QDialog, QWidget, QGridLayout,QMessageBox, QLabel, QPushButton,QLineEdit, QSpinBox,QComboBox,QHBoxLayout
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap
from detecto import core, utils, visualize


class main_window(QtWidgets.QMainWindow):

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
        
        #Usar botón Cargar imagen
        self.button_load = self.findChild(QtWidgets.QPushButton, 'load_button')
        self.button_load.clicked.connect(self.openFileNamesDialog)
        #self.initUI()

        #Usar botón de ejecutar
        self.button_exec = self.findChild(QtWidgets.QPushButton, 'exec_button')
        self.button_exec.clicked.connect(self.model)
        

        #Abrir archivo y mostrar imagen
    def openFileNamesDialog(self):
        #Abrir archivo
        image = QFileDialog.getOpenFileName(None,'OpenFile','',"Image file(*.jpg)")
        self.imagePath = image[0]
        self.image = image
        pixmap = QPixmap(self.imagePath)
        #Escalar archivo
        pixmap2 = pixmap.scaled(190, 140)
        #Imprimir imagen en pantalla
        self.show_image.setPixmap(pixmap2)

        

            
    def closeEvent(self,event):
      reply =QMessageBox.question(self, "Mensaje", "Seguro quiere salir", QMessageBox.Yes, QMessageBox.No)
   
      if reply == QMessageBox.Yes:
       event.accept()
      else: 
        event.ignore()

    def model (self, image):
        model = core.Model.load('avocado_weights.pth', ['fit_avocado', 'unfit_avocado'])
        # Specify the path to your image
        image = utils.read_image(self.imagePath)
        predictions = model.predict(image)
        
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
    
    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = main_window()
    main.show()

    main.setWindowTitle('Bienvenido')
    app.exec_()