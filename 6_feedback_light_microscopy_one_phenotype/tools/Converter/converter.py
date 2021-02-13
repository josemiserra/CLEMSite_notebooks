#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# AUTHOR: Jose Miguel Serra Lleti, EMBL 
# converter.py 
# To further usage, use manual supplied together with CLEMSite

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
####################
import sys
import os.path
from os import listdir,path,getenv,makedirs
import glob
import shutil
import re
import json

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

class Ui_Converter(QMainWindow):
    #####################
    m_samples_folder = ""
    m_destiny_folder = ""
    m_common_name = "image"
    profiles = []

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

    #############
    def setupUi(self, Converter):
        Converter.setObjectName(_fromUtf8("Converter"))
        screen = QDesktopWidget().screenGeometry()
        Converter.resize(screen.width() * 800.0 / 1920.0, screen.height() * 980.0 / 1200.0)
        self.centralwidget = QWidget(Converter)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.buttonBox = QDialogButtonBox(self.centralwidget)
        self.buttonBox.setGeometry(QtCore.QRect(18, screen.width() *900/1920.0, screen.height() *312/ 1200.0, 46))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(18, 18, screen.width() *697/1920, screen.height()*851/1200.0))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0,0,0,0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_3 = QLabel(self.layoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout.addWidget(self.label_3)
        self.lineEdit_profile = QLineEdit(self.layoutWidget)
        self.lineEdit_profile.setObjectName(_fromUtf8("lineEdit_profile"))
        self.horizontalLayout.addWidget(self.lineEdit_profile)
        self.toolButton_profile = QToolButton(self.layoutWidget)
        self.toolButton_profile.setObjectName(_fromUtf8("toolButton_profile"))
        self.horizontalLayout.addWidget(self.toolButton_profile)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.line_2 = QFrame(self.layoutWidget)
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.gridLayout.addWidget(self.line_2, 1, 0, 1, 1)
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.label_11 = QLabel(self.layoutWidget)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.horizontalLayout_11.addWidget(self.label_11)
        self.lineEdit_forigin_2 = QLineEdit(self.layoutWidget)
        self.lineEdit_forigin_2.setObjectName(_fromUtf8("lineEdit_forigin_2"))
        self.horizontalLayout_11.addWidget(self.lineEdit_forigin_2)
        self.toolButton_forigin_2 = QToolButton(self.layoutWidget)
        self.toolButton_forigin_2.setObjectName(_fromUtf8("toolButton_forigin_2"))
        self.horizontalLayout_11.addWidget(self.toolButton_forigin_2)
        self.gridLayout.addLayout(self.horizontalLayout_11, 2, 0, 1, 1)
        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.label_Status_2 = QLabel(self.layoutWidget)
        self.label_Status_2.setObjectName(_fromUtf8("label_Status_2"))
        self.horizontalLayout_12.addWidget(self.label_Status_2)
        self.lineEdit_fdestiny_2 = QLineEdit(self.layoutWidget)
        self.lineEdit_fdestiny_2.setObjectName(_fromUtf8("lineEdit_fdestiny_2"))
        self.horizontalLayout_12.addWidget(self.lineEdit_fdestiny_2)
        self.toolButton_fdestiny_2 = QToolButton(self.layoutWidget)
        self.toolButton_fdestiny_2.setObjectName(_fromUtf8("toolButton_fdestiny_2"))
        self.horizontalLayout_12.addWidget(self.toolButton_fdestiny_2)
        self.gridLayout.addLayout(self.horizontalLayout_12, 3, 0, 1, 1)
        self.line = QFrame(self.layoutWidget)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.gridLayout.addWidget(self.line, 4, 0, 1, 1)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_common = QLabel(self.layoutWidget)
        self.label_common.setObjectName(_fromUtf8("label_common"))
        self.horizontalLayout_4.addWidget(self.label_common)
        self.lineEdit_common = QLineEdit(self.layoutWidget)
        self.lineEdit_common.setObjectName(_fromUtf8("lineEdit_common"))
        self.horizontalLayout_4.addWidget(self.lineEdit_common)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_4, 5, 0, 1, 1)
        self.groupBox = QGroupBox(self.layoutWidget)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout_3 = QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.lineEdit_re = QLineEdit(self.groupBox)
        self.lineEdit_re.setObjectName(_fromUtf8("lineEdit_re"))
        self.gridLayout_3.addWidget(self.lineEdit_re, 5, 1, 1, 1)
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.label_8 = QLabel(self.groupBox)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.horizontalLayout_8.addWidget(self.label_8)
        self.comboBox_4 = QComboBox(self.groupBox)
        self.comboBox_4.setObjectName(_fromUtf8("comboBox_4"))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.horizontalLayout_8.addWidget(self.comboBox_4)
        self.gridLayout_3.addLayout(self.horizontalLayout_8, 1, 0, 1, 1)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.label_9 = QLabel(self.groupBox)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_9.addWidget(self.label_9)
        self.spinBox_2 = QSpinBox(self.groupBox)
        self.spinBox_2.setObjectName(_fromUtf8("spinBox_2"))
        self.horizontalLayout_9.addWidget(self.spinBox_2)
        self.gridLayout_3.addLayout(self.horizontalLayout_9, 2, 0, 1, 1)
        spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem1, 1, 1, 1, 1)
        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.pushButton_Add = QPushButton(self.groupBox)
        self.pushButton_Add.setObjectName(_fromUtf8("pushButton_Add"))
        self.horizontalLayout_10.addWidget(self.pushButton_Add)
        spacerItem2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem2)
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_10.addWidget(self.label_2)
        self.gridLayout_3.addLayout(self.horizontalLayout_10, 5, 0, 1, 1)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_7.addWidget(self.label_7)
        self.comboBox_3 = QComboBox(self.groupBox)
        self.comboBox_3.setObjectName(_fromUtf8("comboBox_3"))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.horizontalLayout_7.addWidget(self.comboBox_3)
        self.gridLayout_3.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        self.checkBox_separated = QCheckBox(self.groupBox)
        self.checkBox_separated.setObjectName(_fromUtf8("checkBox_separated"))
        self.gridLayout_3.addWidget(self.checkBox_separated, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 6, 0, 1, 1)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.pushButton_Run = QPushButton(self.layoutWidget)
        self.pushButton_Run.setObjectName(_fromUtf8("pushButton_Run"))
        self.horizontalLayout_2.addWidget(self.pushButton_Run)
        self.progressBar_2 = QProgressBar(self.layoutWidget)
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName(_fromUtf8("progressBar_2"))
        self.horizontalLayout_2.addWidget(self.progressBar_2)
        self.pushButton_Save = QPushButton(self.layoutWidget)
        self.pushButton_Save.setObjectName(_fromUtf8("pushButton_Save"))
        self.horizontalLayout_2.addWidget(self.pushButton_Save)
        self.gridLayout.addLayout(self.horizontalLayout_2, 8, 0, 1, 1)
        self.label_status = QLabel(self.layoutWidget)
        self.label_status.setObjectName(_fromUtf8("label_status"))
        self.gridLayout.addWidget(self.label_status, 9, 0, 1, 1)
        self.listWidget = QListWidget(self.layoutWidget)
        self.listWidget.setObjectName(_fromUtf8("listWidget"))
        self.gridLayout.addWidget(self.listWidget, 7, 0, 1, 1)
        Converter.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Converter)
        self.menubar.setGeometry(QtCore.QRect(0, 0, screen.height()*734/1900, 38))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        Converter.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Converter)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Converter.setStatusBar(self.statusbar)

        self.retranslateUi(Converter)
        self.toolButton_forigin_2.clicked.connect(self.setupSampleFolder)
        self.toolButton_fdestiny_2.clicked.connect(self.setupDestinyFolder)
        self.pushButton_Run.clicked.connect(self.run)
        self.buttonBox.accepted.connect(self.close)
        self.buttonBox.rejected.connect(self.close)
        self.pushButton_Add.clicked.connect(self.addProfile)
        self.pushButton_Save.clicked.connect(self.saveProfile)
        self.toolButton_profile.clicked.connect(self.loadProfile)


        QtCore.QMetaObject.connectSlotsByName(Converter)

    def retranslateUi(self, Converter):
        Converter.setWindowTitle(_translate("Converter", "Converter ", None))
        self.label_3.setText(_translate("Converter", "Load  profile:", None))
        self.toolButton_profile.setText(_translate("Converter", "...", None))
        self.label_11.setText(_translate("Converter", "Samples folder: ", None))
        self.toolButton_forigin_2.setText(_translate("Converter", "...", None))
        self.label_Status_2.setText(_translate("Converter", "Destination folder: ", None))
        self.toolButton_fdestiny_2.setText(_translate("Converter", "...", None))
        self.label_common.setText(_translate("Converter", "Common image name :", None))
        self.groupBox.setTitle(_translate("Converter", "Add folder", None))
        self.label_8.setText(_translate("Converter", "Lens :", None))
        self.comboBox_4.setItemText(0, _translate("Converter", "5x", None))
        self.comboBox_4.setItemText(1, _translate("Converter", "10x", None))
        self.comboBox_4.setItemText(2, _translate("Converter", "20x", None))
        self.comboBox_4.setItemText(3, _translate("Converter", "40x", None))
        self.label_9.setText(_translate("Converter", "Zoom factor: ", None))
        self.pushButton_Add.setText(_translate("Converter", "Add", None))
        self.label_2.setText(_translate("Converter", "Regular expression:", None))
        self.label_7.setText(_translate("Converter", "Channel:", None))
        self.comboBox_3.setItemText(0, _translate("Converter", "GFP", None))
        self.comboBox_3.setItemText(1, _translate("Converter", "RFP", None))
        self.comboBox_3.setItemText(2, _translate("Converter", "YFP", None))
        self.comboBox_3.setItemText(3, _translate("Converter", "mCherry", None))
        self.comboBox_3.setItemText(4, _translate("Converter", "Hoescht", None))
        self.comboBox_3.setItemText(5, _translate("Converter", "DAPI", None))
        self.comboBox_3.setItemText(6, _translate("Converter", "FITC", None))
        self.comboBox_3.setItemText(7, _translate("Converter", "CFP", None))
        self.comboBox_3.setItemText(8, _translate("Converter", "Alexa_488", None))
        self.comboBox_3.setItemText(9, _translate("Converter", "TexasRed", None))
        self.comboBox_3.setItemText(10, _translate("Converter", "TL -Transmited Light", None))
        self.comboBox_3.setItemText(11, _translate("Converter", "RL - Reflected Light", None))
        self.comboBox_3.setItemText(12, _translate("Converter", "CY3", None))
        self.comboBox_3.setItemText(13, _translate("Converter", "Alexa_568", None))
        self.checkBox_separated.setText(_translate("Converter", "Saved in separated folder", None))
        self.pushButton_Run.setText(_translate("Converter", "Run", None))
        self.pushButton_Save.setText(_translate("Converter", "Save profile", None))
        self.label_status.setText(_translate("Converter", "Status : Empty", None))

    def loadProfile(self):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open file', ".",
                                                  "File experiment (*.prf *.)")
        self.lineEdit_profile.setText(fname)
        if (fname == ''):
            return
        with open(fname) as data_file:
            try:
                self.profiles = json.load(data_file)
            except ValueError as err:
                print(err)
        sep = "_"
        for el in self.profiles:
            if el["separated"]==True:
                sep = "separated"
            else:
                sep = "_"
            self.listWidget.addItem(el["name"] + "::" + el["re"]+"::"+sep)

        self.label_status.setText("Status: Profile loaded.")

    def saveProfile(self):
        fileName = str(QFileDialog.getSaveFileName(self, 'Save LM sequence profile', QtCore.QDir.currentPath(),
                                                         "File experiment .prf (*.prf *.)"))
        if (fileName == ''):
            return
        with open(fileName, 'w') as data_file:
            roi = json.dump(self.profiles, data_file, indent=4, sort_keys=True)

    def setupSampleFolder(self):
        fname = QFileDialog.getExistingDirectory(self, "Find Files", QtCore.QDir.currentPath(),
                                                       QFileDialog.ShowDirsOnly)
        if not fname:
            ret = QMessageBox.warning(self, "Warning",
                                            '''The folder you typed does not exist.\n Select a new one''',
                                            QMessageBox.Ok);
            return
        self.m_samples_folder = fname
        self.lineEdit_forigin_2.setText(fname)

    def setupDestinyFolder(self):
        fname = QFileDialog.getExistingDirectory(self, "Find destiny folder", QtCore.QDir.currentPath(),
                                                       QFileDialog.ShowDirsOnly)
        if not os.path.exists(fname):
            ret = QMessageBox.warning(self, "Warning",
                                            '''The folder you typed doesnt exist.\n Select a new one''',
                                            QMessageBox.Ok);
            return

        self.m_destiny_folder = fname
        self.lineEdit_fdestiny_2.setText(fname)

    def addProfile(self):
        profile_sample = {}
        if (len(self.lineEdit_re.text()) > 2):
            profile_sample["re"] = ".*" + str(self.lineEdit_re.text()) + ".*"
        else:
            profile_sample["re"] = ".*"
        if (len(self.lineEdit_common.text()) > 2):
            self.m_common_name = str(self.lineEdit_common.text())
        profile_sample["name"] = self.m_common_name + "--LM--" + str(self.comboBox_3.currentText()) + "--" + str(
            self.comboBox_4.currentText()) + "--z" + str(
            self.spinBox_2.value())
        self.profiles.append(profile_sample)
        profile_sample["separated"] = self.checkBox_separated.isChecked()
        if self.checkBox_separated.isChecked():
            sep = 'separated'
        else:
            sep = '_'
        self.listWidget.addItem(profile_sample["name"] + "::" + profile_sample["re"]+"::"+sep)

    def run(self):
        # List directories from _0001
        if (self.m_samples_folder == ""):
            self.label_status.setText("Give samples folder first.")
            return

        dfolder = self.m_destiny_folder
        if not dfolder:
            self.label_status.setText("Status: Please, give a folder where to copy the renamed files.")
            return

        flist = os.listdir(str(self.m_samples_folder))
        if not flist:
            self.label_status.setText('Status: Folder of pictures not found.')
            return
        else:
            self.label_status.setText('Status: Directories found.')

        items = []
        for index in range(self.listWidget.count()):
            items.append(str(self.profiles[index]["name"]))
        if (not items):
            self.label_status.setText("Status: Please, add at least one profile.")
            return
        labels = [str(i) for i in items]

        i = 0
        for el in flist:
            i = i + 1
            self.progressBar_2.setValue(int((i * 1.0 / len(flist)) * 100))
            # find all tiffs in the sample
            sfolder = str(self.m_samples_folder) + "\\" + el
            tifs = self.findTiffs(sfolder)
            # create a folder for the sample
            headf, tailf = os.path.split(el)
            dfolder_sample_copy = ""
            s_number = "_" + (4 - len(str(i))) * '0' + str(i)
            if (tifs):
                try:
                    nel = tailf + s_number
                    dfolder_sample_copy = dfolder + "\\" + nel
                except OSError as e:
                    print(e.errno)
                    self.label_status.setText(e.strerror)
                    return
            else:
                continue
            # copy the tiff file there
            k = 0
            tifs = sorted(tifs)
            # For each regular expression in my list get the first picture
            for prof in self.profiles:
                if (k < len(labels)):
                    reg_exp = prof["re"]
                    k = k + 1
                    t = 0
                    for atif in tifs:
                        pos = re.match(reg_exp, atif, flags=0)
                        if (pos):
                            t = t + 1

                            head, tail = os.path.split(atif)
                            nname = prof["name"]
                            pos = re.search("--LM", nname, flags=0)
                            pos2 = re.search("--0[0-9][0-9].ome.tif",tail,flags=0)
                            if(pos2):
                                nname = nname[:pos.start()] + s_number + tail[pos2.start():pos2.start()+5] +  "_" + str(t)+ nname[pos.start():]
                                dfolder_sample_copy_s2 = dfolder_sample_copy+tail[pos2.start():pos2.start()+5]
                            else:
                                nname = nname[:pos.start()] + s_number + "_" + str(t) + nname[pos.start():]
                                dfolder_sample_copy_s2 = dfolder_sample_copy

                            if (not os.path.isdir(dfolder_sample_copy_s2)):
                                os.mkdir(dfolder_sample_copy_s2)

                            if prof["separated"] == True:
                                folder_name = prof["name"]
                                dfolder_sample_copy_s2 = dfolder_sample_copy_s2 + "\\" + folder_name
                                if (not os.path.isdir(dfolder_sample_copy_s2)):
                                    os.mkdir(dfolder_sample_copy_s2)

                            shutil.copy2(atif, str(dfolder_sample_copy_s2))
                            dfolder_samplef = dfolder_sample_copy_s2 + "\\" + tail
                            new_name = dfolder_sample_copy_s2 + "\\" + nname + ".tif"
                            os.rename(dfolder_samplef, new_name)
                else:
                    break

            self.label_status.setText("Status: All files copied and renamed")

    def openMenu(self, position):
        menu = QMenu()
        quitAction = menu.addAction("Delete Value")
        action = menu.exec_(self.listWidget.mapToGlobal(position))
        if action == quitAction:
            listItems = self.listWidget.selectedItems()
            if not listItems:
                return
            else:
                self.listWidget.takeItem(self.listWidget.currentRow())

    def findTiffs(self, foldname):
        flist = []
        tiff_list = []
        dirs = []
        dirs.append(foldname)
        while dirs:
            fname = dirs.pop()
            fname_tif = str(fname) + '\\*.tif*'
            flist = glob.glob(fname_tif)
            if not flist:
                newdirs = ([f for f in glob.glob(fname + '\\*') if os.path.isdir(f)])
                for el in newdirs:
                    dirs.append(el)
            else:
                for el in flist:
                    tiff_list.append(el)
        return tiff_list


def main():
    app = QApplication(sys.argv)

    main = Ui_Converter()
    main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()