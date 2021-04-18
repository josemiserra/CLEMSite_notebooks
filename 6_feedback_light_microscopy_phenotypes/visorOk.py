# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\JMS\Documents\msite\MSite\msiteSEM\UI\register.ui'
#
# Created: Mon Sep 19 20:46:22 2016
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!


# TODO
# Fix B&C unbalance when move slider (recover after switching of channels)
# Adjust if images with differences sizes are selected
# First picture passed is not the one being shown
# Zoom in doesn't work
#########
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os,sys
import cv2
import numpy as np

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig)


def autoBC(img):
    """
    Based on Fiji's auto BC
    :param img:
    :return:
    """
    autoThreshold = 5000
    pixelCount = img.shape[0]*img.shape[1]
    limit = int(pixelCount / 10)
    histogram,bins = np.histogram(img.flatten(),256,[0,256])
    threshold = int(pixelCount / autoThreshold)  # int

    i = -1
    found = False
    count = 0  # int
    while True:
        i += 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
        if found or i >= 255:
            #   if 0 not in (!found, i<255) :
            break
    hmin = i  # int

    i = 256
    while True:
        i -= 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
        if found or i < 0:
            #   if 0 not in (!found, i<255) :
            break
    hmax = i  # int

    #norm_img = np.zeros(img.shape)
    #cv2.normalize(img, norm_img, hmin, hmax, norm_type=cv2.NORM_MINMAX )
    nimg = np.array((((img-hmin) / (hmax - hmin))*255),dtype = np.uint8)
    nimg[img > hmax] = 255
    nimg[img < hmin] = 0
    return nimg


class Ui_VisorWindow(QtWidgets.QDialog):
    shift = []
    m_equalize = []
    _zoom = 0
    factor = 1
    pixelsize_ref = 0
    o_chR = []
    o_chG = []
    o_chB = []
    chR =[]
    chG = []
    chB = []
    imhide = [False, False, False]
    def __init__(self):
        self.channel_BC = np.zeros((3,2),dtype = np.int)
        self.channel_BC[:,1] = 10
        self.no_images = True
        super(Ui_VisorWindow, self).__init__()
        self.setupUi(self)



    def setListImages(self,images_path_list, values):
        if not images_path_list:
            return
        self.values = values
        path, header = os.path.split(images_path_list[0])
        self.path = path
        self.tag = header[:-4]
        self.images_path_list = images_path_list
        image = cv2.imread(images_path_list[0])
        self.o_chR = image[:,:,0]
        self.o_chG = image[:,:,1]
        self.o_chB = image[:,:,2]
        self.chR = self.o_chR.copy()
        self.chG = self.o_chG.copy()
        self.chB = self.o_chB.copy()
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()

        self.comboBox_list.blockSignals(True)
        self.comboBox_image_1.blockSignals(True)
        self.comboBox_image_2.blockSignals(True)
        self.comboBox_image_3.blockSignals(True)
        for ind in range(3):
            self.comboBox_image_1.addItem(_fromUtf8(""))
            self.comboBox_image_1.setItemText(ind, "channel "+str(ind))
            self.comboBox_image_2.addItem(_fromUtf8(""))
            self.comboBox_image_2.setItemText(ind, "channel "+str(ind))
            self.comboBox_image_3.addItem(_fromUtf8(""))
            self.comboBox_image_3.setItemText(ind, "channel "+str(ind))
        self.comboBox_list.clear()
        for ind, el in enumerate(images_path_list):
            _, tail = os.path.split(el)
            self.comboBox_list.addItem(_fromUtf8(""))
            self.comboBox_list.setItemText(ind,tail)

        self.comboBox_list.setCurrentIndex(0)
        self.comboBox_image_1.setCurrentIndex(0)
        self.comboBox_image_2.setCurrentIndex(1)
        self.comboBox_image_3.setCurrentIndex(2)
        self.drawImage(self.chR,self.chG, self.chB)
        self.comboBox_list.blockSignals(False)
        self.comboBox_image_1.blockSignals(False)
        self.comboBox_image_2.blockSignals(False)
        self.comboBox_image_3.blockSignals(False)
        self.SpinIm.blockSignals(True)
        self.SpinIm.setMaximum(len(self.images_path_list)-1)
        self.SpinIm.setValue(0)
        self.SpinIm.blockSignals(False)
        self.labelValue.setText("Value :"+self.values[0])
        self.setupForm()
        self.repaint()

    def setupUi(self, VisorWindow):
        VisorWindow.setObjectName(_fromUtf8("VisorWindow"))
        VisorWindow.resize(1218, 1265)
        screen = QDesktopWidget().screenGeometry()
        VisorWindow.resize(screen.width() * 1200.0 / 2880.0, screen.height() * 1250.0 / 1620.0)
        VisorWindow.setWindowFlags(VisorWindow.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint)

        self.gridLayout_2 = QGridLayout(VisorWindow)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.pushButton = QPushButton(VisorWindow)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.gridLayout_2.addWidget(self.pushButton, 4, 0, 1, 1)
        self.comboBox_list = QComboBox(VisorWindow)
        self.comboBox_list.setObjectName(_fromUtf8("comboBox_list"))
        self.gridLayout_2.addWidget(self.comboBox_list, 4, 1, 1, 1)
        self.SpinIm = QSpinBox(VisorWindow)
        self.SpinIm.setMaximum(1)
        self.SpinIm.setSingleStep(1)
        self.SpinIm.setObjectName(_fromUtf8("SpinIm"))
        self.gridLayout_2.addWidget(self.SpinIm, 4, 2, 1, 1)
        self.label_4 = QLabel(VisorWindow)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_2.addWidget(self.label_4, 1, 2, 1, 1)
        self.label_5 = QLabel(VisorWindow)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_2.addWidget(self.label_5, 2, 2, 1, 1)
        self.horizontalSlider_c = QSlider(VisorWindow)
        self.horizontalSlider_c.setMaximum(100)
        self.horizontalSlider_c.setMinimum(0)
        self.horizontalSlider_c.setSingleStep(10)
        self.horizontalSlider_c.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_c.setObjectName(_fromUtf8("horizontalSlider_c"))
        self.gridLayout_2.addWidget(self.horizontalSlider_c, 2, 1, 1, 1)
        self.formLayout = QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QLabel(VisorWindow)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)
        self.doubleSpinBoxX = QDoubleSpinBox(VisorWindow)
        self.doubleSpinBoxX.setMaximum(360.0)
        self.doubleSpinBoxX.setSingleStep(0.5)
        self.doubleSpinBoxX.setObjectName(_fromUtf8("doubleSpinBoxX"))
        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.doubleSpinBoxX)
        self.pushButton_flipH = QPushButton(VisorWindow)
        self.pushButton_flipH.setObjectName(_fromUtf8("pushButton_flipH"))
        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.pushButton_flipH)
        self.pushButton_flipV = QPushButton(VisorWindow)
        self.pushButton_flipV.setObjectName(_fromUtf8("pushButton_flipV"))
        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.pushButton_flipV)
        self.gridLayout_2.addLayout(self.formLayout, 1, 0, 2, 1)
        self.horizontalSlider_b = QSlider(VisorWindow)
        self.horizontalSlider_b.setMaximum(50)
        self.horizontalSlider_b.setMinimum(-50)
        self.horizontalSlider_b.setSingleStep(10)
        self.horizontalSlider_b.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_b.setInvertedAppearance(False)
        self.horizontalSlider_b.setInvertedControls(False)
        self.horizontalSlider_b.setObjectName(_fromUtf8("horizontalSlider_b"))
        self.gridLayout_2.addWidget(self.horizontalSlider_b, 1, 1, 1, 1)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.gridLayout_2.addLayout(self.horizontalLayout, 4, 1, 1, 2)
        self.graphicsView = QGraphicsView(VisorWindow)
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.gridLayout_2.addWidget(self.graphicsView, 0, 0, 1, 3)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.checkBox_c1 = QCheckBox(VisorWindow)
        self.checkBox_c1.setText(_fromUtf8(""))
        self.checkBox_c1.setObjectName(_fromUtf8("checkBox_c1"))
        self.horizontalLayout_5.addWidget(self.checkBox_c1)
        self.button_c1_auto = QPushButton(VisorWindow)
        self.button_c1_auto.setText(_fromUtf8(""))
        self.button_c1_auto.setObjectName(_fromUtf8("button_c1_auto"))

        self.horizontalLayout_5.addWidget(self.button_c1_auto)
    
        self.comboBox_image_1 = QComboBox(VisorWindow)
        self.comboBox_image_1.setObjectName(_fromUtf8("comboBox_image_1"))
        self.horizontalLayout_5.addWidget(self.comboBox_image_1)

        self.checkBox_equalize_1 = QCheckBox(VisorWindow)
        self.checkBox_equalize_1.setObjectName(_fromUtf8("checkBox_equalize_1"))
        self.horizontalLayout_5.addWidget(self.checkBox_equalize_1)
        self.gridLayout.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.checkBox_c2 = QCheckBox(VisorWindow)
        self.checkBox_c2.setText(_fromUtf8(""))
        self.checkBox_c2.setObjectName(_fromUtf8("checkBox_c2"))
        self.horizontalLayout_3.addWidget(self.checkBox_c2)
        self.button_c2_auto = QPushButton(VisorWindow)
        self.button_c2_auto.setText(_fromUtf8(""))
        self.button_c2_auto.setObjectName(_fromUtf8("button_c2_auto"))
        self.horizontalLayout_3.addWidget(self.button_c2_auto)

        self.comboBox_image_2 = QComboBox(VisorWindow)
        self.comboBox_image_2.setObjectName(_fromUtf8("comboBox_image_2"))
        self.horizontalLayout_3.addWidget(self.comboBox_image_2)
        self.checkBox_equalize_2 = QCheckBox(VisorWindow)
        self.checkBox_equalize_2.setObjectName(_fromUtf8("checkBox_equalize_2"))
        self.horizontalLayout_3.addWidget(self.checkBox_equalize_2)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.checkBox_c3 = QCheckBox(VisorWindow)
        self.checkBox_c3.setText(_fromUtf8(""))
        self.checkBox_c3.setObjectName(_fromUtf8("checkBox_c3"))
        self.horizontalLayout_4.addWidget(self.checkBox_c3)
        self.button_c3_auto = QPushButton(VisorWindow)
        self.button_c3_auto.setText(_fromUtf8(""))
        self.button_c3_auto.setObjectName(_fromUtf8("button_c3_auto"))
        self.horizontalLayout_4.addWidget(self.button_c3_auto)

        self.comboBox_image_3 = QComboBox(VisorWindow)
        self.comboBox_image_3.setObjectName(_fromUtf8("comboBox_image_3"))
        self.horizontalLayout_4.addWidget(self.comboBox_image_3)
        self.checkBox_equalize_3 = QCheckBox(VisorWindow)
        self.checkBox_equalize_3.setObjectName(_fromUtf8("checkBox_equalize_3"))
        self.horizontalLayout_4.addWidget(self.checkBox_equalize_3)
        self.gridLayout.addLayout(self.horizontalLayout_4, 2, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 3, 0, 1, 3)

        self.labelValue = QLabel(VisorWindow)
        self.labelValue.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.labelValue, 4, 0, 1, 1)

        self._zoom = 0
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.graphicsView.setScene(self._scene)
        self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphicsView.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.graphicsView.setFrameShape(QFrame.NoFrame)


        self.retranslateUi(VisorWindow)

        self.checkBox_c1.stateChanged[int].connect(self.redraw)
        self.checkBox_c2.stateChanged[int].connect(self.redraw)
        self.checkBox_c3.stateChanged[int].connect(self.redraw)

        self.button_c1_auto.clicked.connect(self.OnAutoC0)
        self.button_c2_auto.clicked.connect(self.OnAutoC1)
        self.button_c3_auto.clicked.connect(self.OnAutoC2)

        self.checkBox_equalize_1.stateChanged[int].connect(self.redraw)
        self.checkBox_equalize_2.stateChanged[int].connect(self.redraw)
        self.checkBox_equalize_3.stateChanged[int].connect(self.redraw)

        self.horizontalSlider_b.valueChanged[int].connect(self.OnChangeBrightnessContrast)
        self.horizontalSlider_c.valueChanged[int].connect(self.OnChangeBrightnessContrast)
        self.doubleSpinBoxX.valueChanged[float].connect(self.OnChangeRotation)

        self.pushButton.clicked.connect(self.savePic)
        self.pushButton_flipH.clicked.connect(self.OnFlipH)

        self.pushButton_flipV.clicked.connect(self.OnFlipV)

        self.comboBox_image_1.currentIndexChanged.connect(self.OnChange_image1)
        self.comboBox_image_2.currentIndexChanged.connect(self.OnChange_image2)
        self.comboBox_image_3.currentIndexChanged.connect(self.OnChange_image3)

        self.SpinIm.valueChanged.connect(self.imageChanged)

        self.comboBox_list.currentIndexChanged.connect(self.OnChange_image)
        QtCore.QMetaObject.connectSlotsByName(VisorWindow)



    def retranslateUi(self, VisorWindow):
        VisorWindow.setWindowTitle(_translate("VisorWindow", "VisorWindow", None))
        self.pushButton.setText(_translate("VisorWindow", "Save", None))
        self.label_4.setText(_translate("VisorWindow", "B", None))
        self.label_5.setText(_translate("VisorWindow", "C", None))
        self.label.setText(_translate("VisorWindow", "Rotate:", None))
        self.labelValue.setText(_translate("VisorWindow", "Value:", None))
        self.pushButton_flipH.setText(_translate("VisorWindow", "Flip H", None))
        self.pushButton_flipV.setText(_translate("VisorWindow", "Flip V", None))
        self.checkBox_equalize_1.setText(_translate("VisorWindow", "Equalize", None))
        self.checkBox_equalize_2.setText(_translate("VisorWindow", "Equalize", None))
        self.checkBox_equalize_3.setText(_translate("VisorWindow", "Equalize", None))
        self.button_c1_auto.setText(_translate("VisorWindow", "Auto", None))
        self.button_c2_auto.setText(_translate("VisorWindow", "Auto", None))
        self.button_c3_auto.setText(_translate("VisorWindow", "Auto", None))
        self.checkBox_c1.setChecked(True)
        self.checkBox_c2.setChecked(True)
        self.checkBox_c3.setChecked(True)
        self.horizontalSlider_c.setValue(10)



    def closeEvent(self, event):
        event.accept()  # let the window close

    def drawImage(self,imR,imG=None,imB=None):

        overlay = np.zeros(shape=imR.shape + (3,), dtype=np.uint8)

        if (self.checkBox_c1.isChecked()):
            if (self.button_c1_auto.isChecked()):
                imR = autoBC(imR)
            if (self.checkBox_equalize_1.isChecked()):
                imR = cv2.equalizeHist(imR)
            if self.imhide[0] :
                imR = self.chR
                self.imhide[0] = False
            overlay[..., 0] = imR
        else:
            self.imhide[0] = True

        if(np.any(imG)):
            if (self.checkBox_c2.isChecked()):
                if (self.button_c2_auto.isChecked()):
                    imG = autoBC(imG)
                if (self.checkBox_equalize_2.isChecked()):
                    imG = cv2.equalizeHist(imG)
                if self.imhide[1]:
                    imG = self.chG
                    self.imhide[1] = False
                overlay[..., 1] = imG
            else:
                self.imhide[1]=True

        if(np.any(imB)):
            if (self.checkBox_c3.isChecked()):
                if (self.button_c3_auto.isChecked()):
                    imB = autoBC(imB)
                if (self.checkBox_equalize_3.isChecked()):
                    imB = cv2.equalizeHist(imB)
                if self.imhide[2]:
                    imB = self.chB
                    self.imhide[2] = False
                overlay[..., 2] = imB
            else:
                self.imhide[2] = True


        self.currentImagePath = self.path + "\\tmp.jpg"
        cv2.imwrite(self.currentImagePath, overlay)
        self.setPhoto(QPixmap(self.currentImagePath))
        if (self.factor > 0):
            self.graphicsView.scale(self.factor, self.factor)
        self.currentimage = overlay


    def OnAutoC0(self):
        self.resetBC()
        self.chR =autoBC(self.chR)
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()


    def OnAutoC1(self):
        self.resetBC()
        self.chG = autoBC(self.chG)
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()


    def OnAutoC2(self):
        self.resetBC()
        self.chB = autoBC(self.chB)
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()


    def OnChangeBrightnessContrast(self):
        self.updateBC()

    def updateBC(self):
        if len(self.images_path_list) == 0:
            return
        self.channel_BC[0,0]= int(self.horizontalSlider_b.value())
        self.channel_BC[0,1] = int(self.horizontalSlider_c.value())
        chR = cv2.add(self.chR,int(self.horizontalSlider_b.value()))
        chR = cv2.multiply(chR,(self.horizontalSlider_c.value()/10.0))

        self.channel_BC[1, 0] = int(self.horizontalSlider_b.value())
        self.channel_BC[1, 1] = int(self.horizontalSlider_c.value())
        chG = cv2.add(self.chG, int(self.horizontalSlider_b.value()))
        chG = cv2.multiply(chG,(self.horizontalSlider_c.value()/10.0))


        self.channel_BC[2, 0] = int(self.horizontalSlider_b.value())
        self.channel_BC[2, 1] = int(self.horizontalSlider_c.value())
        chB = cv2.add(self.chB, int(self.horizontalSlider_b.value()))
        chB = cv2.multiply(chB,(self.horizontalSlider_c.value()/10.0))
        self.drawImage(chR, chG, chB)
        self.fitInView()
        return


    def savePic(self,):
        # Create dir
        directory = '.\\selection'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for el in self.images_path_list:
            img = cv2.imread(el)
            fname = self.path+directory+'\\'+self.comboBox_list.currentText()
            cv2.imwrite(str(fname), img)

    def OnChangeChannelBC(self):
        if (self.checkBox_c1.isChecked()):
            self.horizontalSlider_b.setValue(self.channel_BC[0, 0])
            self.horizontalSlider_c.setValue(self.channel_BC[0, 1])
        elif (self.checkBox_c2.isChecked()):
            self.horizontalSlider_b.setValue(self.channel_BC[1, 0])
            self.horizontalSlider_c.setValue(self.channel_BC[1, 1])
        elif (self.checkBox_c3.isChecked()):
            self.horizontalSlider_b.setValue(self.channel_BC[2, 0])
            self.horizontalSlider_c.setValue(self.channel_BC[2, 1])

    def resetBC(self):
        if (self.checkBox_c1.isChecked()):
            self.channel_BC[0, 0] = 0
            self.channel_BC[0, 1] = 10
        elif (self.checkBox_c2.isChecked()):
            self.channel_BC[1, 0] = 0
            self.channel_BC[1, 1] = 10
        elif (self.checkBox_c3.isChecked()):
            self.channel_BC[2, 0] = 0
            self.channel_BC[2, 1] = 10
        self.OnChangeChannelBC()


    def zoomFactor(self):
        return self._zoom

    def wheelEvent(self, event):
        if not self._photo.pixmap().isNull():

            numDegrees = event.angleDelta()/8
            if(numDegrees):
                numSteps = numDegrees.y()/15
                if numSteps > 0:
                    factor = 1.25
                    self._zoom += 1
                else:
                    factor = 0.8
                    self._zoom -= 1
                if self._zoom > 0:
                    self.graphicsView.scale(factor, factor)
                elif self._zoom == 0:
                    self.fitInView()
                else:
                    self._zoom = 0

    def setupForm(self):
        pass
        #self.setWindowTitle(self.tag)

    def setPhoto(self, pixmap=None):

        if pixmap and not pixmap.isNull():
            self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self.graphicsView.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())

    def fitInView(self):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            unity = self.graphicsView.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.graphicsView.scale(1 / unity.width(), 1 / unity.height())
            # viewrect = self.graphicsView.viewport().rect()
            # scenerect = self.graphicsView.transform().mapRect(rect)
            # factor = min(viewrect.width() / scenerect.width(),
            #             viewrect.height() / scenerect.height())
            # self.graphicsView.scale(factor, factor)
            self.graphicsView.centerOn(rect.center())
            self._zoom = 0

    def redraw(self):
        image = cv2.imread(self.currentImagePath)
        self.drawImage(image[:,:,0], image[:,:,1], image[:,:,2])
        self.fitInView()

    def OnChangeRotation(self):
        self.graphicsView.resetTransform()
        self.graphicsView.rotate(float(self.doubleSpinBoxX.value()))

    def OnFlipV(self):
        self.chB = cv2.flip(self.chB, 0)
        self.chR = cv2.flip(self.chR, 0)
        self.chG = cv2.flip(self.chG, 0)
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()

    def OnFlipH(self):
        self.chB = cv2.flip(self.chB, 1)
        self.chR = cv2.flip(self.chR, 1)
        self.chG = cv2.flip(self.chG, 1)
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()

    def OnChange_image(self):
        self.graphicsView.resetTransform()
        image = cv2.imread(self.images_path_list[int(self.comboBox_list.currentIndex())])
        self.o_chR = image[:,:,0]
        self.o_chG = image[:,:,1]
        self.o_chB = image[:,:,2]
        self.chR = self.o_chR.copy()
        self.chG = self.o_chG.copy()
        self.chB = self.o_chB.copy()
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()
        self.updateBC()

    def OnChange_image1(self):
        self.graphicsView.resetTransform()
        if self.comboBox_image_1.currentIndex() == 1 :
            self.chR =  self.o_chG
        elif self.comboBox_image_1.currentIndex() == 2 :
            self.chR = self.o_chB
        elif self.comboBox_image_1.currentIndex() == 0 :
            self.chR = self.o_chR
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()
        self.updateBC()
    def OnChange_image2(self):
        self.graphicsView.resetTransform()
        if self.comboBox_image_2.currentIndex() == 0:
            self.chG = self.o_chR
        elif self.comboBox_image_2.currentIndex() == 2:
            self.chG = self.o_chB
        elif self.comboBox_image_2.currentIndex() == 1 :
            self.chG = self.o_chG
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()
        self.updateBC()
    def OnChange_image3(self):
        self.graphicsView.resetTransform()
        if self.comboBox_image_3.currentIndex() == 0:
            self.chB = self.o_chR
        elif self.comboBox_image_3.currentIndex() == 1:
            self.chB = self.o_chG
        elif self.comboBox_image_3.currentIndex() == 2 :
            self.chB = self.o_chB
        self.drawImage(self.chR, self.chG, self.chB)
        self.fitInView()
        self.updateBC()

    def imageChanged(self):
        ind = self.SpinIm.value()
        self.labelValue.setText("Value :"+self.values[ind])
        self.comboBox_list.setCurrentIndex(ind)

import argparse
# defined command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()
CLI.add_argument("--listImages",
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=None,)  # default if nothing is provided
CLI.add_argument( "--folder",  # name on the CLI - drop the `--` for positional/required parameters
  type=str,
  default=None,)  # default if nothing is provided
CLI.add_argument( "--listIndex",  # name on the CLI - drop the `--` for positional/required parameters
  nargs='*',
  type=str,
  default=None,)
CLI.add_argument( "--listValues",  # name on the CLI - drop the `--` for positional/required parameters
  nargs='*',
  type=str,
  default=None,)
if __name__ == "__main__":
    # app = QApplication(sys.argv)
    args = CLI.parse_args()
    dfolder = os.getcwd()
    list_images = []
    list_values = []
    if args.folder:
        dfolder = args.folder.strip('\'')
    print('Show :')
    print(args.listValues)
    for i,el in enumerate(args.listImages):
        el = el.strip('.[],\'')
        value = args.listValues[i]
        value = value.strip('[],\'')
        list_values.append(value)
        list_images.append(os.path.normpath(dfolder+el))
        ind = args.listIndex[i].strip('.[],\'')
        print("- " + str(ind) + " : " + str(el))
    try:
        visor.setListImages(list_images,list_values)
    except NameError:
        visor = Ui_VisorWindow()
        visor.setListImages(list_images, list_values)

    visor.show()
    #sys.exit(app.exec_())

