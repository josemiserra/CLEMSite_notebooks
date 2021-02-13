import sys
from PyQt4 import QtGui, QtCore
from coordinategeneratorGUI import Ui_Dialog as Dlg
import cam_communicator_class as cc
import pdb
import os
import numpy as np
import leica_mark_and_find_helper as mf
import global_settings as gs
import generate_coords as gc
import time

class GeneratorDialog(QtGui.QDialog, Dlg):
    #############################
    #
    #   GUI STUFF
    #
    #############################

    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.setupUi(self)
        self.debugging = False
        self.coords = np.zeros((4,3))
        self.coords[:] = np.nan
        self.coords[:,2] = 1.0
        # display global settings
        self.tl_IP.setText("IP address: " + gs.ip)
        self.tl_spotdistance.setText("Spot distance: " + str(gs.spotdistance[0])+","+str(gs.spotdistance[1]))
        self.tl_spotnumber.setText("Spot number: " + str(gs.spotnumber[0])+","+str(gs.spotnumber[1]))
        self.tl_subposdistance.setText("Subpos distance: " + str(gs.subpos_distance[0])+","+str(gs.subpos_distance[1]))
        self.tl_subposnumber.setText("Subpos number: " + str(gs.subpos_nr[0])+","+str(gs.subpos_nr[1]))
        # connect to Leica Microscope
        self.CAMC = cc.CAMcommunicator()
        self.CAMC.setIP(gs.ip)
        self.CAMC.open()
        time.sleep(3)
        print "flushing bugger"
        self.CAMC.flushCAMreceivebuffer()
        time.sleep(3)
        print "getting current stage position"
        print self.CAMC.getCurrentStagePosition()

        # connect buttons to callbacks
        self.pb_TL.clicked.connect(self.clickTL)
        self.pb_TR.clicked.connect(self.clickTR)
        self.pb_BL.clicked.connect(self.clickBL)
        self.pb_BR.clicked.connect(self.clickBR)
        self.pb_generatelist_centre.clicked.connect(self.gen_positionlist_centres_only)
        self.pb_generatelist_subpos.clicked.connect(self.gen_positionlist)

        print "performing gui update"
        self.guiUpdate()
        #self.pushButton_disconnect.clicked.connect(self.disconnectCAM)
        #self.pushButton_learn.clicked.connect(self.learn)

        #self.setLayout(layout)
        #buttons = {}
        #for i in range(24):
        #    for j in range(16):
        #        # keep a reference to the buttons
        #        text = chr(j+ord('A')) + str(i+1).zfill(2)
        #        buttons[(i, j)]  = QtGui.QPushButton(text)
        #        # add to the layout
        #        buttons[(i, j)].clicked.connect(self.movestage)
        #        self.wellgrid.addWidget(buttons[( i,j)], j, i)

    ######################
    # core stuff
    ######################

    def clickTL(self):
        print "clickTopLeft"
        if self.debugging:
            self.coords[0,0] = 0.0
            self.coords[0,1] = 1.0
        else:
           self.coords[0,:] = self.CAMC.getCurrentStagePosition()
           print self.coords[0,:]
        self.guiUpdate()

    def clickTR(self):
        print "clickTopRight"
        if self.debugging:
            self.coords[1,0] = 1.0
            self.coords[1,1] = 1.0
        else:
            self.coords[1,:] = self.CAMC.getCurrentStagePosition()
            print self.coords[1,:]
        self.guiUpdate()

    def clickBL(self):
        print "clickBottomLeft"
        if self.debugging:
            self.coords[2,0] = 0.0
            self.coords[2,1] = 0.0
        else:
            self.coords[2,:] = self.CAMC.getCurrentStagePosition()
            print self.coords[2,:]
        self.guiUpdate()

    def clickBR(self):
        print "clickBottomRight"
        if self.debugging:
            self.coords[3,0] = 1.0
            self.coords[3,1] = 0.0
        else:
            self.coords[3,:] = self.CAMC.getCurrentStagePosition()
            print self.coords[3,:]
        self.guiUpdate()

    def guiUpdate(self):
        print "triggering gui update"

        for i,label in enumerate([self.tl_TL, self.tl_TR, self.tl_BL, self.tl_BR]):
            label.setText(str(self.coords[i,0])+","+str(self.coords[i,1]))
        if np.any(np.isnan(self.coords)):
            print "grey out action buttons"
            self.pb_generatelist_centre.setEnabled(False)
            self.pb_generatelist_subpos.setEnabled(False)
        else:
            self.pb_generatelist_centre.setEnabled(True)
            self.pb_generatelist_subpos.setEnabled(True)
            print "enable action buttons"

    def gen_positionlist(self):
        print "in generate position list"
        pts = gc.generate_all_positions()
        H = gc.calculate_transform(self.coords)
        pts_transformed = pts.copy()
        for i, pt in  enumerate(pts):
            pts_transformed[i]=gc.transferPoint(H, pt[0], pt[1])
        print pts_transformed
        tmp = QtGui.QFileDialog.getSaveFileName(self, 'Save as', '.')

        #self.show()
        # the next line is a fix as QFileDialog sometimes returns paths with / seperator instead of \\ on win
        tmp = str(tmp).replace("/",os.sep)
        mf.write_mark_and_find_list(tmp, pts_transformed)


    def gen_positionlist_centres_only(self):
        print "in generate centre positions only"

    def connectCAM(self):
        #tmp =str(self.lineEdit_IP.text())
        #ipaddr = tmp.strip()
        #self.printConsole("Connecting to CAM server " + ipaddr)
        #self.CAMC.setIP(ipaddr)
        #self.CAMC.open()
        pass

    def disconnectCAM(self):
        pass
        #self.printConsole("Disconnecting from CAM server.")
        #self.CAMC.close()

    def learn(self):
        pass
        #self.printConsole ("Querying current stage position")

        #if tmp is not None:
        #    self.TopLeftCoord = np.array(tmp).view(layout.StagePosition)
        #    self.TopLeftCoord[2]= -0.00075  # maximum height when Super-Z is enabled
        #    self.label_xpos.setText(str(100*self.TopLeftCoord[0]))
        #    self.label_ypos.setText(str(100*self.TopLeftCoord[1]))
        #    self.wellpattern = self.layout_factory.createRegularArray("well384", self.TopLeftCoord )
        #    self.printConsole( "Setting top left position of 384well plate to [m]: (" + str(self.TopLeftCoord[0]) + ',' + str(self.TopLeftCoord[1]) + "," + str(self.TopLeftCoord[2]) + ")" )
        #    print self.wellpattern
        #else:
        #    self.printConsole( "Error querying stage position" )

    # def printConsole(self,str):
    #     self.textEdit.append(str)
    #     self.textEdit.moveCursor(QtGui.QTextCursor.End)
    #     QtGui.QApplication.processEvents()
    #
    # def movestage(self):
    #     sender = self.sender()
    #     wellname = sender.text()
    #     print "Going to Well ", wellname
    #
    #     xcoord = int(str(wellname)[1:3])-1
    #     ycoord = ord(str(wellname)[0])-ord('A')
    #
    #     self.printConsole("selected well " + str(wellname) + ".  x:" +  str(xcoord)  +", y: " + str(ycoord))
    #
    #     if self.wellpattern is not None:
    #         pos= self.wellpattern[xcoord,ycoord]
    #         self.printConsole("Moving stage to position (" + str(pos[0]) + "," + str(pos[1]) + "," + str(pos[2]) + ")")
    #         self.CAMC.setStagePosition(pos)
    #         time.sleep(3)
    #         self.printConsole("Done.\n")

####
# Setup main dialog
####
app = QtGui.QApplication(sys.argv)
dialog = GeneratorDialog()
dialog.show()
sys.exit(app.exec_())
