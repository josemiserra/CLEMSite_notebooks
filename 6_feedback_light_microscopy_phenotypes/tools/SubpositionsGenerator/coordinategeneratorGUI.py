# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'coordinategeneratorGUI.ui'
#
# Created: Mon Feb 29 16:25:41 2016
#      by: PyQt4 UI code generator 4.9.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(698, 387)
        self.pb_TL = QtGui.QPushButton(Dialog)
        self.pb_TL.setGeometry(QtCore.QRect(30, 50, 221, 32))
        self.pb_TL.setObjectName(_fromUtf8("pb_TL"))
        self.pb_BR = QtGui.QPushButton(Dialog)
        self.pb_BR.setGeometry(QtCore.QRect(290, 150, 221, 32))
        self.pb_BR.setObjectName(_fromUtf8("pb_BR"))
        self.pb_TR = QtGui.QPushButton(Dialog)
        self.pb_TR.setGeometry(QtCore.QRect(290, 50, 221, 32))
        self.pb_TR.setObjectName(_fromUtf8("pb_TR"))
        self.pb_BL = QtGui.QPushButton(Dialog)
        self.pb_BL.setGeometry(QtCore.QRect(30, 150, 221, 32))
        self.pb_BL.setObjectName(_fromUtf8("pb_BL"))
        self.pb_generatelist_centre = QtGui.QPushButton(Dialog)
        self.pb_generatelist_centre.setGeometry(QtCore.QRect(30, 250, 291, 21))
        self.pb_generatelist_centre.setObjectName(_fromUtf8("pb_generatelist_centre"))
        self.pb_generatelist_subpos = QtGui.QPushButton(Dialog)
        self.pb_generatelist_subpos.setGeometry(QtCore.QRect(30, 290, 291, 21))
        self.pb_generatelist_subpos.setObjectName(_fromUtf8("pb_generatelist_subpos"))
        self.tl_TL = QtGui.QLabel(Dialog)
        self.tl_TL.setGeometry(QtCore.QRect(40, 30, 201, 21))
        self.tl_TL.setObjectName(_fromUtf8("tl_TL"))
        self.tl_BL = QtGui.QLabel(Dialog)
        self.tl_BL.setGeometry(QtCore.QRect(40, 130, 201, 21))
        self.tl_BL.setObjectName(_fromUtf8("tl_BL"))
        self.tl_TR = QtGui.QLabel(Dialog)
        self.tl_TR.setGeometry(QtCore.QRect(300, 30, 201, 21))
        self.tl_TR.setObjectName(_fromUtf8("tl_TR"))
        self.tl_BR = QtGui.QLabel(Dialog)
        self.tl_BR.setGeometry(QtCore.QRect(300, 130, 201, 21))
        self.tl_BR.setObjectName(_fromUtf8("tl_BR"))
        self.tl_TL_2 = QtGui.QLabel(Dialog)
        self.tl_TL_2.setGeometry(QtCore.QRect(360, 210, 281, 21))
        self.tl_TL_2.setObjectName(_fromUtf8("tl_TL_2"))
        self.tl_IP = QtGui.QLabel(Dialog)
        self.tl_IP.setGeometry(QtCore.QRect(360, 240, 281, 21))
        self.tl_IP.setObjectName(_fromUtf8("tl_IP"))
        self.tl_spotdistance = QtGui.QLabel(Dialog)
        self.tl_spotdistance.setGeometry(QtCore.QRect(360, 260, 281, 21))
        self.tl_spotdistance.setObjectName(_fromUtf8("tl_spotdistance"))
        self.tl_spotnumber = QtGui.QLabel(Dialog)
        self.tl_spotnumber.setGeometry(QtCore.QRect(360, 290, 281, 21))
        self.tl_spotnumber.setObjectName(_fromUtf8("tl_spotnumber"))
        self.tl_subposdistance = QtGui.QLabel(Dialog)
        self.tl_subposdistance.setGeometry(QtCore.QRect(360, 320, 281, 21))
        self.tl_subposdistance.setObjectName(_fromUtf8("tl_subposdistance"))
        self.tl_subposnumber = QtGui.QLabel(Dialog)
        self.tl_subposnumber.setGeometry(QtCore.QRect(360, 350, 281, 21))
        self.tl_subposnumber.setObjectName(_fromUtf8("tl_subposnumber"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_TL.setText(QtGui.QApplication.translate("Dialog", "get top left stage pos ", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_BR.setText(QtGui.QApplication.translate("Dialog", "get bottom right stage pos ", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_TR.setText(QtGui.QApplication.translate("Dialog", "get top right stage pos ", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_BL.setText(QtGui.QApplication.translate("Dialog", "get bottom left stage pos ", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_generatelist_centre.setText(QtGui.QApplication.translate("Dialog", "generate position list (center positions)", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_generatelist_subpos.setText(QtGui.QApplication.translate("Dialog", "generate position list (subpositions)", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_TL.setText(QtGui.QApplication.translate("Dialog", "Not set", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_BL.setText(QtGui.QApplication.translate("Dialog", "Not set", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_TR.setText(QtGui.QApplication.translate("Dialog", "Not set", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_BR.setText(QtGui.QApplication.translate("Dialog", "Not set", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_TL_2.setText(QtGui.QApplication.translate("Dialog", "Fixed settings (edit global_settings.py) :", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_IP.setText(QtGui.QApplication.translate("Dialog", "IP", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_spotdistance.setText(QtGui.QApplication.translate("Dialog", "spotdistance", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_spotnumber.setText(QtGui.QApplication.translate("Dialog", "spotnumber", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_subposdistance.setText(QtGui.QApplication.translate("Dialog", "subposdistance", None, QtGui.QApplication.UnicodeUTF8))
        self.tl_subposnumber.setText(QtGui.QApplication.translate("Dialog", "subposnumber", None, QtGui.QApplication.UnicodeUTF8))

