import global_settings as gs
import pylab
import numpy as np
from find_affine import *
from scipy import linalg
from PyQt4.QtCore import pyqtRemoveInputHook
import pdb

def generate_generic_grid(nx,ny, offset=(0,0), spacing=(1.0,1.0)):
    # generates an generic position list for
    # a grid with nx*ny postions spaced by spacing and
    # located at offset
    xpos = np.arange(nx)*spacing[0] + offset[0]
    ypos = np.arange(ny)*spacing[1] + offset[1]
    xx,yy = np.meshgrid(xpos,ypos)
    return np.hstack((xx.reshape(nx*ny,1), yy.reshape(nx*ny,1)))

def generate_all_positions():
    spotpositions = generate_generic_grid(gs.spotnumber[0],gs.spotnumber[1], spacing=gs.spotdistance)
    tmp = np.zeros((0,2))
    for spot in spotpositions:
        print  "Spot ",  spot
        print  "offset ", -np.array(gs.subpos_nr)/2.0+0.5
        print  "distance ",gs.subpos_distance
        #pdb.set_trace()
        subpos = generate_generic_grid(gs.subpos_nr[0], gs.subpos_nr[1], spot-(np.array(gs.subpos_nr)/2.0-0.5)*gs.subpos_distance, gs.subpos_distance)
        print "Subpos "< subpos
        print "xxxxxxxx"
        tmp = np.vstack((tmp,subpos))
    return tmp

def transferPoint(H, x, y, inverse=False):
     # transfers a single point using a Homography
     tmp = np.float32([x,y,1.0])
     if not inverse:
         trh = np.dot(H, tmp.transpose())
     else:
         trh = np.dot(linalg.inv(H), tmp.transpose())
     trh /= trh[2]
     return(trh[0:2])

def calculate_transform(stage_corners=None):
    grid_corners  = np.zeros((4, 3), dtype=np.float32)

    #TL
    grid_corners[0,0] = 0
    grid_corners[0,1] = 0
    grid_corners[0,2] = 1

    # TR
    grid_corners[1,0] = gs.spotdistance[0]*(gs.spotnumber[0]-1)
    grid_corners[1,1] = 0
    grid_corners[1,2] = 1

    # BL
    grid_corners[2,0] = 0
    grid_corners[2,1] = gs.spotdistance[1]*(gs.spotnumber[1]-1)
    grid_corners[2,2] = 1

    # BR
    grid_corners[3,0] = gs.spotdistance[0]*(gs.spotnumber[0]-1)
    grid_corners[3,1] = gs.spotdistance[1]*(gs.spotnumber[1]-1)
    grid_corners[3,2] = 1

    if stage_corners is None:
        stage_corners = np.zeros((4, 3), dtype=np.float32)
        #TL
        stage_corners[0,0] = 0
        stage_corners[0,1] = 0
        stage_corners[0,2] = 1
        # TR
        stage_corners[1,0] = 1
        stage_corners[1,1] = 0
        stage_corners[1,2] = 1
        # BL
        stage_corners[2,0] = 0
        stage_corners[2,1] = 1
        stage_corners[2,2] = 1
        # BR
        stage_corners[3,0] = 1
        stage_corners[3,1] = 1
        stage_corners[3,2] = 1

    print grid_corners
    print stage_corners

    #for i in range(len(self.coords.pickedx)):
    #        grid_corners[i,0]=self.coords.pickedx[i]
    #        grid_corners[i,1]=self.coords.pickedy[i]
    #        stage[i,0]= self.coords.stageCoords[i][0]
    #        stage[i,1]= self.coords.stageCoords[i][1]
    #    print(grid_corners)
    #    print(stage)
    H= Haffine_from_points(grid_corners.transpose(),stage_corners.transpose())
    return(H)

def test_transform():
    pts, centrepts = generate_all_positions() # requires modification to return centrepoints
    #pylab.plot(pts[:,0],pts_transfor[:,1],"*")
    #pylab.plot(centrepts[:,0], centrepts_transformed[:,1],"+")
    H = calculate_transform()
    pts_transformed = pts.copy()
    for i, pt in  enumerate(pts):
        pts_transformed[i]=transferPoint(H, pt[0], pt[1])
    centrepts_transformed = centrepts.copy()
    for i, pt in  enumerate(centrepts):
        centrepts_transformed[i]=transferPoint(H, pt[0], pt[1])
    print "Transformed Points:"
    print pts_transformed
    pylab.plot(pts_transformed[:,0],pts_transformed[:,1],"*")
    pylab.plot(centrepts_transformed[:,0], centrepts_transformed[:,1],"+")
    pylab.show()