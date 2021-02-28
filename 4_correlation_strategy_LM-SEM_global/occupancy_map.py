#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLEMSite . This software was build for correlative microscopy using FIB SEM.

Simple occupancy-grid-based mapping without localization. This map is a legacy, only a few features from it are still used.

Scale is 1 for mm, 1000 for micrometers, and so on...

# @Title			: Map
# @Project			: CLEMSite
# @Description		: Software for correlation 
# @Author			: Jose Miguel Serra Lleti
# @Email			: lleti (at) embl.de
# @Copyright		: Copyright (C) 2018 Jose Miguel Serra Lleti
# @License			: MIT Licence
# @Developer		: Jose Miguel Serra Lleti
# 					  EMBL, Cell Biology and Biophysics
# 					  Department of Molecular Structural Biology
# @Date				: 2020/03
# @Version			: 1.2
# @Python_version	: 3.6
"""
# ======================================================================================================================


import numpy as np
import os
import math
from stack import Stack
import csv
from scipy.spatial import KDTree,distance
from inspect import getsourcefile

class ZMap(object):
      position = []
      z_value = []


class Map(object):
    """
    The Map class stores an occupancy grid as a two dimensional
    numpy array.

    Public instance variables:

        width      --  Number of columns in the occupancy grid.
        height     --  Number of rows in the occupancy grid.
        resolution --  Width of each grid square in meters.
        origin_x   --  Position of the grid cell (0,0) in
        origin_y   --    in the map coordinate system.
        grid       --  numpy array with height rows and width columns.


    Note that x increases with increasing column number and y increases
    with increasing row number.
    """
    GAUSSIAN_WEIGHTS = {'1': [1,2], '2': [2,2], '3': [3,3], '4':[4,4], '5': [5,5], '6': [6,6],
             '7': [8,8], '8': [9,9], '9': [11,11], '10': [12,12]}

    ## Additional information useful to build the virtual mesh
    DIST_SQ = 0.56  # mm, 600 um
    DIST_SPACE = 0.02  # 20
    DIST_GRID = 0.6
    scale = 1 # for mm
    pixelSize = 1
    orientation = 0

    label_occupation_map = dict()
    fixed_points_map = dict()
    map_labels = dict()
    map_cf = dict()
    map_coordinates_origin = dict()
    map_coordinates_destiny = dict()
    spacing = 10
    colors_list =  ['darkred','blue','darkgreen','orange','dimgray']
    cols = 20
    rows = 20

    def __init__(self, scale, preferences= None, origin_x=0, origin_y=0, resolution=.1,
                 width=50, height=50, orientation = 0):
        """ Construct an empty occupancy grid.

        Arguments: origin_x,
                   origin_y  -- The position of grid cell (0,0) in the
                                map coordinate frame.
                   resolution-- width and height of the grid cells
                                in meters.
                   width,
                   height    -- The grid will have height rows and width
                                columns cells.  width is the size of
                                the x-dimension and height is the size
                                of the y-dimension.
                    type -- If EM, the calculated coordinates are mirrored (flipped)

         The default arguments put (0,0) in the center of the grid.

        """

        self.spacing = 10
        self.blocksize = 40

        self.scale = 1
        self.origin_x = origin_x
        self.origin_y = origin_y
        # Get path of this script
        wk_dir, _ = os.path.split(os.path.abspath(getsourcefile(lambda: 0)))
        #self.template_picture = wk_dir+'/dialogs/res/default_grid.png'

        self.resolution = resolution
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.visited = np.zeros((height, width))
        self.orientation = orientation


        if preferences is not None:
            self.configureMap(preferences)
        else:
            from MsiteHelper import readPreferences
            preferences = readPreferences( wk_dir+'\\user.pref')
            self.configureMap(preferences)

    @staticmethod
    def getMapCoordinates(lnames):
        xlabels = '0123456789abcdefghijk+'
        ylabels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ*'
        rows = len(lnames)
        mapcoordinates = np.zeros((rows, 2))
        for k in range(rows):
            cletter = lnames[k]
            poslminus = xlabels.find(cletter[0])
            poslplus = ylabels.find(cletter[1])
            ext_x = 0
            ext_y = 0
            if (len(cletter) > 2):
                if cletter[4] == 1:
                    ext_x = -1
                    ext_y = -1
                elif cletter[4] == 2:
                    ext_x = +1
                    ext_y = -1
                elif cletter[4] == 4:
                    ext_x = -1
                    ext_y = +1
                elif cletter[4] == 3:
                    ext_x = +1
                    ext_y = +1
            mapcoordinates[k, 0] = poslminus * 10 + ext_x
            mapcoordinates[k, 1] = poslplus * 10 + ext_y
        return mapcoordinates

    def configureMap(self, preferences):
        """
            Takes the preferences from the project, stored in CLEMSITE
            and loads the parameters of the grid by default.
            The idea is in the future to refactor the class to adapt
            to any type of grid or shape, by reading everything from 
            preferences (blocksize, spacing, labels)

        """
        self.template_picture = preferences['grid'][0]['template_file']
        self.blocksize = int(preferences['grid'][0]['blocksize'])
        self.spacing = int(preferences['grid'][0]['spacing'])
        self.xlabels_or = "0123456789abcdefghijk+"  # Use them as reference
        self.ylabels_or = "ABCDEFGHIJKLMNOPQRSTUVWXYZ*"

        self.xlabels = "0123456789abcdefghijk+"
        self.ylabels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ*"

        self.label_occupation_map = [[0 for x in range(len(self.xlabels_or))] for x in
                                     range(len(self.ylabels_or))]  # grid map of completed samples.s
        self.cols = len(self.xlabels)  # 16
        self.rows = len(self.ylabels)  # 16 is a good sizes

        self.x = np.arange(self.cols)
        self.y = np.arange(self.rows)
        ctx = 0
        cty = 0
        for i in self.x:
            for j in self.y:
                mlabel = self.xlabels[ctx] + self.ylabels[cty]
                c1 = self.xlabels.index(mlabel[0])
                c2 = self.ylabels.index(mlabel[1])
                self.map_labels[mlabel] = np.float32([c1 * self.spacing, c2 * self.spacing, 0.0])
                self.map_cf[mlabel] = False;
                self.map_coordinates_origin[mlabel] = np.array([0.0, 0.0], dtype=np.float32)
                self.map_coordinates_destiny[mlabel] = np.array([0.0, 0.0], dtype=np.float32)
                cty = (cty + 1) % self.rows
            ctx = (ctx + 1) % self.cols

    def getLabels(self):
        return list(self.map_labels.keys())

    def flatten(self):
        # Flatten the numpy array into a list of integers from 0-100.
        # This assumes that the grid entries are probalities in the
        # range 0-1. This code will need to be modified if the grid
        # entries are given a different interpretation (like
        # log-odds).
        flat_grid = self.grid.reshape((self.grid.size,)) * 100
        self.grid_data = list(np.round(flat_grid))
        return self.grid_data

    def set_cell(self, x, y, val):
        """ Set the value of a cell in the grid.

        Arguments:
            x, y  - This is a point in the map coordinate frame.
            val   - This is the value that should be assigned to the
                    grid cell that contains (x,y).

        """
        self.grid[x,y] = val

    def getXlabels(self):
        return self.xlabels

    def getYlabels(self):
        return self.ylabels

    def setXlabels(self, nxl):
        """Need for reset labels in X axis"""
        self.xlabels = nxl
        self.max_xticks = len(nxl)

    def setYlabels(self, nxy):
        """Need for reset labels in Y axis"""
        self.ylabels = nxy
        self.max_yticks = len(nxy)

    def getTag(self,cx,cy):
        if cx > len(self.xlabels) or cy > len(self.ylabels) or cx < 0 or cy < 0 :
            return "_"
        return self.xlabels[int(cx / 10)] + self.ylabels[int(cy / 10)] + "_" + str(
            int((((cx / 10.0) - math.floor(cx / 10.0)) * 10) / 2)) + str(
            int((((cy / 10) - math.floor(cy / 10)) * 10) / 2))

    def find_closest_letter(self, coords,map_value):
        """
        Given a set of 2D coordinates in the plane, it gives you back the closest landmark
        """
        coords = np.array(coords,dtype=np.float32)
        if(coords.shape[0]==2):
            coords = np.array((coords[0],coords[1],0.0),dtype = np.float32)
        if (map_value == 1):
            distance, index = KDTree(self.map_coordinates_destiny.values()).query(coords)
            mlist = list(self.map_coordinates_destiny.keys())
            label = mlist[index]
        else:
            distance, index = KDTree(self.map_coordinates_origin.values()).query(coords)
            mlist = list(self.map_coordinates_origin.keys())
            label = mlist[index]

        return label,distance

    def find_square_letter(self, coords):
        """
        Given a set of 2D coordinates in the plane, it gives you back the closest landmark
        """
        for key, value in self.map_labels.items():
                if (value[0] == math.floor(coords[0] / 10) * 10 and value[1] == math.floor(coords[1] / 10) * 10):
                    letter = key
                    return letter

    def generateGridMapCoordinates(self,datalm,itags,orientation, type ='LM'):

        self.orientation = orientation
        cpoint = datalm[0]
        letter = itags[0]

        ixlabels = []
        iylabels = []
        if type == 'EM':
            ixlabels = self.xlabels[::-1]
            iylabels = self.ylabels

        # The idea now is go to each one of the 4 neighbors
        # The central point is then taken and now we are going to take the
        # direction vector for each point
        # check orthogonality between points
        distgrid = np.zeros((2,1))
        distgrid[0] = self.DIST_GRID
        distgrid[1] = self.DIST_GRID

        # This tries to get the distance from the data itself
        if(datalm.shape[0]>1):
            good_points = []
            good_distances = []
            M = distance.pdist(datalm)
            M  = distance.squareform(M)
            list_good_points = []
            for i in range(M.shape[0]):
                if((self.DIST_GRID+self.DIST_GRID*0.1) > M[0][i] > (self.DIST_GRID-self.DIST_GRID*0.2)):
                            good_points.append(datalm[i])
                            good_distances.append(M[0][i])
            if(len(good_distances)>1):
                # The point with less distance in x, is distX
                m_list_x= []
                for el in good_points:
                    m_list_x.append(distance.euclidean(cpoint[0],el[0]))
                inde_x =np.argmin(np.array(m_list_x))
                distgrid[0] = good_distances[inde_x]
                m_list_y = []
                for el in good_points:
                    m_list_y.append(distance.euclidean(cpoint[1], el[1]))
                inde_y = np.argmin(np.array(m_list_y))
                distgrid[1] = good_distances[inde_y]
        self.calculated_distgrid = distgrid


        gstack = Stack()
        gstack.push(letter)
        self.map_cf[letter] = True
        self.map_coordinates_destiny[letter] = np.array([cpoint[0], cpoint[1], 0.0], dtype=np.float32)
        self.map_coordinates_origin[letter] = self.map_labels[letter]
        count = 0
        while (not gstack.isEmpty() or count < 3):
            # get first element
            count += 1
            fp = gstack.pop()
            cpoint = self.map_coordinates_destiny[fp]
            tags, neighs = self.calculate4neighbors(fp, cpoint,distgrid, ixlabels, iylabels, self.orientation)
            for let, val in zip(tags, neighs):
                if self.map_cf[let] == False:
                    if (not (let[0] == '+' or let[1] == '*' or let[0] == '*' or let[1] == '+')):
                        gstack.push(let)
                        cpoint = np.array([val[0], val[1], 0.0], dtype=np.float32)
                        self.map_coordinates_destiny[let] = cpoint
                        self.map_coordinates_origin[let] = self.map_labels[let]
                        self.map_cf[let] = True
        self.map_exists = True
        all_keys = list(self.map_coordinates_destiny.keys())

        for key in all_keys:
            if (np.all(self.map_coordinates_destiny[key] == 0)):
                self.map_coordinates_origin.pop(key, None)
                self.map_coordinates_destiny.pop(key, None)
            else:
                self.fixed_points_map[key] = False

    def update_grid(self,points,tags):
        for el1, el2 in zip(points, tags):
            self.map_coordinates_destiny[el2] = np.array([el1[0], el1[1], 0.0], dtype=np.float32)
            self.fixed_points_map[el2] = True  # True values
    
    def update_neighborhood(self,point,tag):
        # TODO:
        # when add new point to the occupancy map this point computes the error between what is expected
        # and what is the point itself
        expected = self.map_coordinates_destiny[tag]
        error_d  = distance(expected,point)
        # the error how much we need to propagate. The bigger the error, the bigger the propagation

    def get4Neighbors(self, letter):
        # Map exists already
        ntags = []
        neighs = []

        ctx = self.xlabels.index(letter[0])
        cty = self.ylabels.index(letter[1])
        if (ctx - 1 < 0 or cty - 1 < 0 or ctx + 1 > len(self.xlabels) or cty + 1 > len(self.ylabels)):
            return ntags, neighs
        nu = self.xlabels[ctx] + self.ylabels[cty - 1]
        nd = self.xlabels[ctx] + self.ylabels[cty + 1]
        nr = self.xlabels[ctx + 1] + self.ylabels[cty]
        nl = self.xlabels[ctx - 1] + self.ylabels[cty]
        tags =[letter,nu,nd,nr,nl]
        origincs = []
        destinycs =[]
        for el in tags:
            origincs.append(self.map_coordinates_origin[el])
            destinycs.append(self.map_coordinates_destiny[el])
        origincs = np.array(origincs,dtype = np.float32)
        destinycs = np.array(destinycs,dtype = np.float32)
        return tags,origincs,destinycs

    def calculate4neighbors(self, letter, centralpoint, distgrid, gxlabels, gylabels, orientation):

        ntags = []
        neighs = []

        ctx = gxlabels.index(letter[0])
        cty = gylabels.index(letter[1])
        if (ctx - 1 < 0 or cty - 1 < 0 or ctx+1 > len(gxlabels)-1 or cty + 1 > len(gylabels)-1):
            return ntags, neighs
        nu = gxlabels[ctx] + gylabels[cty - 1]
        nd = gxlabels[ctx] + gylabels[cty + 1]
        nr = gxlabels[ctx + 1] + gylabels[cty]
        nl = gxlabels[ctx - 1] + gylabels[cty]

        neighs = []
        ntags = [nl, nu, nr, nd]

        # module = (math.sqrt((centralpoint[0]-el[0])**2+(centralpoint[1]-el[1])**2))
        # if(module>1e-10):
        # mod_el = math.sqrt(el[0]*el[0]+el[1]*el[1])
        # vprod =  centralpoint[0]*el[0]+centralpoint[1]*el[1]
        # cos_alpha =   vprod/(mod_el*mod_cp)
        # sen_alpha = 1-cos_alpha*cos_alpha
        angposr = math.radians(orientation)
        neighs.append([centralpoint[0] - distgrid[1] * math.cos(angposr),
                       centralpoint[1] + distgrid[0] * math.sin(angposr)])
        neighs.append([centralpoint[0] + distgrid[1] * math.sin(angposr),
                       centralpoint[1] + distgrid[0]  * math.cos(angposr)])  # Correct
        neighs.append([centralpoint[0] + distgrid[1]  * math.cos(angposr),
                       centralpoint[1] - distgrid[0]  * math.sin(angposr)])  # Correct
        neighs.append([centralpoint[0] - distgrid[1] * math.sin(angposr),
                       centralpoint[1] - distgrid[0] * math.cos(angposr)])
        return ntags, neighs

    def is_pattern(self, point_id):
        i = self.xlabels_or.find(point_id[0])
        j = self.ylabels_or.find(point_id[1])
        return((i>-1 and j>-1)and(len(point_id)==2))

    def squarify(self,fpoints_filename):
        # Purge to Squares "squarifies" the data incoming from the grid
        # First it will try to find the file fpoints. fpoints is something like this
        # 286,235,1
        # 262,261,1
        # 262,235,1
        # 285,261,1
        #  274,248,1
        # 674,242,2
        # ...
        with open(fpoints_filename, 'rb') as f:
            reader = csv.reader(f)
            points_list = list(reader)

        points_by_group = {}
        for el in points_list:
            coords = np.array([el[0], el[1]], dtype=np.float32)
            if (el[2] not in points_by_group.keys()):
                points_by_group[str(el[2])] = []
            points_by_group[el[2]].append(coords)

        mdist = []
        my_std = []
        for group in points_by_group:
            alist = points_by_group[group]
            alist = alist[0:4]

            mdist.append(math.sqrt((alist[0][0] - alist[1][0]) ** 2 + (alist[0][1] - alist[1][1]) ** 2))
            mdist.append(math.sqrt((alist[1][0] - alist[2][0]) ** 2 + (alist[1][1] - alist[2][1]) ** 2))
            mdist.append(math.sqrt((alist[2][0] - alist[3][0]) ** 2 + (alist[2][1] - alist[3][1]) ** 2))
            mdist.append(math.sqrt((alist[3][0] - alist[0][0]) ** 2 + (alist[3][1] - alist[0][1]) ** 2))
            my_std.append(np.std(np.array(mdist, dtype=np.float32)))

        # We will take then the std closest to 0, in other words, the minimum
        ind = np.argmin(np.array(my_std, dtype=np.float32))
        # Which is the index equivalent to the group.
        return ind

    def getCoordinatesGrid(self,tags_list):
        return [self.map_labels[el] for el in tags_list]

    def getNeighbors(self,cletter, N=1, complete_group = False):
        if complete_group:
            xlabels = '+0123456789abcdefghijk+'
            ylabels = '*ABCDEFGHIJKLMNOPQRSTUVWXYZ*'
        else:
            xlabels = self.xlabels
            ylabels = self.ylabels

        poslminus = xlabels.find(cletter[0])
        poslplus = ylabels.find(cletter[1])
        Nband = int(N)
        neighs = []
        for i in range(poslminus - Nband, poslminus + Nband + 1):
            if len(xlabels) > i > -1:
                minusl = xlabels[i]
                for j in range(poslplus - Nband, poslplus + Nband + 1):
                    if len(ylabels) > j > -1:
                        plusl = ylabels[j]
                        neighs.append(minusl + plusl)
        return neighs

    def getPrevious(self,letter, flip=False):
        xlabels = "0123456789abcdefghijk+"
        ylabels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ*"
        if flip:
            xlabels = xlabels[::-1]
        ctx = xlabels.index(letter[0])
        cty = ylabels.index(letter[1])
        # nu = self.xlabels[ctx] + self.ylabels[cty - 1]
        # nd = self.xlabels[ctx] + self.ylabels[cty + 1]
        #nr = self.xlabels[ctx + 1] + self.ylabels[cty]
        nl = xlabels[ctx - 1] + ylabels[cty]
        return nl

    def getCoordinatesFromLabel(self,mlabel):
        """Given a Label name get back a coordinate for the canvas"""
        ind_x = self.xlabels.index(mlabel[0])
        ind_y = self.ylabels.index(mlabel[1])
        return (ind_x*self.spacing,ind_y*self.spacing)
