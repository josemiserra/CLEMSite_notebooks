# -*- coding: utf-8 -*-
"""
Created on Sat Jun 06 09:49:33 2015

@author: JMS
"""
import random
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from scipy.linalg import orth
from occupancy_map import Map,ZMap
from ptp import LocalArea,PointToPoint,matrixrank, anglebetween

from math import degrees
import json
import threading
from multiprocessing.pool import ThreadPool
from contextlib import closing

import scipy.spatial as spt

class PointType:
        calibrated = "CALIBRATED" # Points that have both map coordinates
        non_calibrated = "NON_CALIBRATED" # Points with map1 coordinates but not with map2.
        target = "TARGET" # Points with map1 but that only can be predicted to map2.
        acquired = "ACQUIRED" # Points with only map2 but with no information about map1
        unknown = "NA"


class State:
    """
        The class State is a special feature that does not correspond to the PointType.
        The PointType is a static situation that gives identity to the point.
        The state is something temporary that can be altered.
    """
    protected = "PROTECTED" # Point has been manually overwritten and cannot be modified
    blocked = "BLOCKED"
    zeroed = "" # No especial states

class virtualGridMap(object):
    """
        A virtual map is a class that gets all the information of the grid and tries
        to give a prediction of unknown positions.
        It considers two homologous maps and establishes correspondences between them.
        E.g.:
            - Given a LM coordinate, returns the corresponding estimation of the SEM (not possible in LM map)
            - Given a letter returns the corresponding coordinates of the estimated center
            - Given a coordinate, estimate the letter where we are going to land


        
       Representation of the points 
       We have selected 4 different kind of points:
           - Non Calibrated NC: points coming from LM without assigned correspondence, used for calibration
           - Calibrated C: points coming from LM, with the correspondent SEM coordinates, used for calibration
           - Targets T: points coming from LM used for targeting
           - Acquisition Acq: points acquired on the fly
           
        Instead of saving the points in 4 different lists, we are saving all of them in one array and then
        saving the indices for each categorie (Ind).
        That allows having points belonging to more than one categorie, or easily to introduce
        more category points.

        Could be a 2D or a 3D
       
     """    
    __metaclass__ = ABCMeta


    warning_transformation =""
    map_lock = threading.Lock()


    def __init__(self,logger, force2D =False, parent = None):
          self.logger = logger

          self.current_pos = "" # Landmark reference
          self.last_point_added = ""
          # LANDMARK
          # Dataframe instead of class reason it is because the
          # porting to a file is immediate and the managing of lists of arrays too.
          # In design terms, having a Landmark class would be much better, but in practical terms
          # slows down. The following is a mixture between class and database, linked by the landmark ID
          self.columns = [ 'LANDMARK','TYPE', 'STATE',
                          'UPDATE_ORIGIN','UPDATE_DESTINY','UPDATE_TAG',
                          'COORDS_ORIGIN_X', 'COORDS_ORIGIN_Y', 'COORDS_ORIGIN_Z',
                          'COORDS_DESTINY_X', 'COORDS_DESTINY_Y', 'COORDS_DESTINY_Z']

          #
          self.rms_avg = []
          self.rms_sd =  []
          self.columns_corigin = ['LANDMARK','BELIEF','COORDS_ORIGIN_X', 'COORDS_ORIGIN_Y', 'COORDS_ORIGIN_Z']
          self.columns_cdestiny =['LANDMARK','BELIEF','COORDS_DESTINY_X', 'COORDS_DESTINY_Y', 'COORDS_DESTINY_Z']

          if(force2D):
            self.col_dim_coords_origin = ['COORDS_ORIGIN_X','COORDS_ORIGIN_Y']
            self.col_dim_coords_destiny = ['COORDS_DESTINY_X','COORDS_DESTINY_Y']
          else:
            self.col_dim_coords_origin = ['COORDS_ORIGIN_X', 'COORDS_ORIGIN_Y','COORDS_ORIGIN_Z']
            self.col_dim_coords_destiny = ['COORDS_DESTINY_X', 'COORDS_DESTINY_Y','COORDS_DESTINY_Z']

          self.col_reset = ['RMS_AVG','RMS_SD']

          self.map_df = pd.DataFrame(columns=self.columns)
          self.cor_df = pd.DataFrame(columns=self.columns_corigin)
          self.cde_df = pd.DataFrame(columns=self.columns_cdestiny)

          self.list_local_area = {}  # every point can have a radius of action

          # List of error associated to each point
          self.list_errorOrigin =  {}
          self.list_errorDestiny = {}

          self.map_exists = False
          self.map_id = "map1_map2"

          self.CalibratedPtp = PointToPoint()
          self.GlobalPtp = PointToPoint()
          # Occupancy map
          self.grid_map = Map(1)
          self.orientation = 0


    @staticmethod
    def dist_microns(x, y):
        return np.sqrt(np.sum((x - y) ** 2)) * 1000.0  ## Error in um

    @staticmethod
    def dist(x, y):
        if (x[0] == np.inf or x[1] == np.inf or y[0] == np.inf or y[1] == np.inf):
            return np.inf
        else:
            return np.sqrt(np.sum((x - y) ** 2))

    def checkValidSystem(self, calculateOrientation = False):
        # Get all calibration points
        coordsOrigin, coordsDestiny, pids = self.getLandmarksByType(PointType.calibrated)
        coordsDestiny = coordsDestiny[:,0:2]
        if(matrixrank(coordsDestiny,1)>=2):
            # TODO : calculate orientation based on data
            # A = orth(coordsDestiny)
            # angle = anglebetween(A[0],[1,0])
            #if(calculateOrientation):
            # self.orientation = np.rad2deg(angle) # this angle has to b
            return True

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        eps = np.finfo(np.float32).eps
        if (np.sum(np.linalg.norm(vector)) < eps):
            return vector
        return vector / np.linalg.norm(vector)


    def collinear(p0, p1, p2):
        x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
        x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
        val = x1 * y2 - x2 * y1
        return abs(val) < 1e-2

    def loadMap(self,dict_map):
        # Split in 3 dictionaries
        stmap = dict_map['MAP']
        stcor = dict_map['COR']
        stcde = dict_map['CDE']

        self.map_df = pd.read_json(stmap)
        self.cor_df = pd.read_json(stcor)
        self.cde_df = pd.read_json(stcde)

        for index, row in self.map_df.iterrows():
            p_id = str(row['LANDMARK'])
            self.list_local_area[p_id] = LocalArea()

    def isEmpty(self,arr):
        arr = np.array(arr)
        if not np.any(arr.shape):
            return True
        if(arr.size == 0):
            return True
        if np.any(np.isinf(arr.astype(float))):
            return True
        return False

    def getTotalLandmarks(self):
        return len(self.map_df)

    def getLandmarkIds(self):
        """
            Return available ids
        """
        return list(self.map_df.LANDMARK);

    def getCoordsFromLandmarks(self,ilids,map_value):
        list_coords = []
        for el in ilids:
            coords = self.getLandmark(el, map_value)
            if(not np.any(np.isinf(coords))):
                list_coords.append(coords)
        return np.array(list_coords)

    def getLandmarksByType(self, type):
        """
            ACK
        """
        df2 = self.map_df.loc[self.map_df['TYPE'] == type]
        point_ids = list(df2['LANDMARK'])
        coordsOrigin = self.getCoordsFromLandmarks(point_ids,1)
        coordsDestiny = self.getCoordsFromLandmarks(point_ids, 2)
        return coordsOrigin,coordsDestiny,point_ids

    def getLandmarkIDsByType(self, type):
        """
            ACK
        """
        df2 = self.map_df.loc[self.map_df['TYPE'] == type]
        point_ids = list(df2['LANDMARK'])
        return point_ids

    def checkState(self,point_id,state):
        df2 = self.map_df.loc[self.map_df['STATE'] == state] # Get all points in state
        return np.any(df2['LANDMARK'].isin([point_id])); # Return true if any of the points is in the list

    def isin(self,point_id):
        return  np.any(self.map_df['LANDMARK'].isin([point_id]));

    def checkType(self,point_id,type):
        df2 = self.map_df.loc[self.map_df['TYPE'] == type] # Get all points by type
        return(np.any(df2['LANDMARK'].isin([point_id]))); # Return true if any of the points is in the list

    def getLandmarkType(self,point_id):
        df2 = self.map_df.loc[self.map_df['LANDMARK']==point_id]
        flist = list(df2['TYPE'])
        return flist[0]

    def getLandmarkState(self,point_id):
        df2 = self.map_df.loc[self.map_df['LANDMARK']==point_id]
        flist = list(df2['STATE'])
        return flist[0]

    def setLandmarkId(self,old_id,new_id):
        """
            ACK

        """
        if(self.isin(old_id)):
            self.map_df.loc[self.map_df['LANDMARK']==old_id,'LANDMARK'] = new_id
            self.cor_df.loc[self.cor_df['LANDMARK']==old_id,'LANDMARK'] = new_id
            self.cde_df.loc[self.cde_df['LANDMARK']==old_id,'LANDMARK'] = new_id
            self.list_local_area[new_id] = self.list_local_area[old_id]
            del self.list_local_area[old_id]
            self.list_errorDestiny[new_id] = self.list_errorDestiny[old_id]
            del self.list_errorDestiny[old_id]
            self.list_errorOrigin[new_id] = self.list_errorOrigin[old_id]
            del self.list_errorOrigin[old_id]
            return "OK"
        else:
            return "ERROR: id not in list"

    def getLandmark(self,point_id,map_value):
        """
                    Map value returns the coordinates : 1 for origin, 2 for destiny

        """
        if(not self.isin(point_id)):
            return np.array([-np.inf])
        if (map_value == 1):
            coords = self.map_df.loc[self.map_df['LANDMARK'] == point_id,self.col_dim_coords_origin]
            coords = np.squeeze(coords.values)
            return np.array(coords,dtype = np.float32)
        elif (map_value == 2):
            coords = self.map_df.loc[self.map_df['LANDMARK'] == point_id, self.col_dim_coords_destiny]
            coords = np.squeeze(coords.values)
            return np.array(coords,dtype = np.float32)
        else:
            self.logger.error("ERROR: In getLandmark for :" + str(point_id) + ". From " + str(self.map_id) + " Use map_value 1 to origin, 2 to destiny.")
        return np.array([-np.inf])



    def updateLandmarks(self):
        """
        Update inner set of landmarks
        :return:
        """

        point_ids = self.getLandmarkIds()
        for el in point_ids:
            self.updateLandmark(el)


    def updateLandmark(self,point_id):
        """
                    Map value returns the coordinates : 1 for origin, 2 for destiny

        """
        if not self.cor_df['LANDMARK'].empty:
            df_pid = self.cor_df.loc[self.cor_df['LANDMARK'] == point_id]

            if not df_pid.empty :
                if len(df_pid) == 1:
                    # UPDATE GENERAL LANDMARK MAP
                    coords = np.array(df_pid[self.col_dim_coords_origin],dtype=np.float32)[0]
                    
                    self.map_df.loc[self.map_df['LANDMARK'] == point_id, self.col_dim_coords_origin] = coords[range(0, len(self.col_dim_coords_origin))]
                    
                else:
                    coords = self.averageLandmarkPosition(np.array(df_pid[self.col_dim_coords_origin],dtype=np.float32), np.array(df_pid['BELIEF']))
                    
                    self.map_df.loc[self.map_df['LANDMARK'] == point_id,self.col_dim_coords_origin] = coords[range(0,len(self.col_dim_coords_origin))]
                    

        if not self.cde_df['LANDMARK'].empty:
            df_pid = self.cde_df.loc[self.cde_df['LANDMARK'] == point_id]

            if not df_pid.empty:
                # UPDATE GENERAL LANDMARK MAP
                if len(df_pid) == 1:
                    coords = np.array(df_pid[self.col_dim_coords_destiny],dtype=np.float32)[0]
                    
                    self.map_df.loc[self.map_df['LANDMARK'] == point_id, self.col_dim_coords_destiny] = coords[range(0, len(self.col_dim_coords_destiny))]
                    
                else:
                    coords =  self.averageLandmarkPosition(np.array(df_pid[self.col_dim_coords_destiny],dtype=np.float32), np.array(df_pid['BELIEF']))
                    
                    self.map_df.loc[self.map_df['LANDMARK'] == point_id, self.col_dim_coords_destiny] = coords[range(0,len(self.col_dim_coords_destiny))]
                    

    def resetCoordinates(self, point_id, map_id):
        """
            Set coordinates to 0
            Map value returns the coordinates : 1 for origin, 2 for destiny

        """

        if (not self.isin(point_id)):
                    return -1
        if map_id == 1:
                self.cor_df = self.cor_df[self.cor_df.LANDMARK != point_id]
                self.addCoordsOrigin(point_id, np.zeros(len(self.col_dim_coords_origin)), 0.0)
                self.list_errorOrigin[point_id] = []
                self.list_local_area[point_id] = LocalArea()
        if map_id == 2:
                self.cde_df = self.cde_df[self.cde_df.LANDMARK != point_id]
                self.addCoordsDestiny(point_id, np.zeros(len(self.col_dim_coords_destiny)), 0.0)
                self.list_errorDestiny[point_id] = []
                self.list_local_area[point_id] = LocalArea()




    def averageLandmarkPosition(self, coords, belief, method = 'average'):
        """

         We are going to start with a simple method of determining the landmark position by averaging all points estimated.

        :param col_names:
        :param df:
        :return:
        """
        if(method=='average'):
            n_arr = (coords.transpose() * belief).transpose() # Multiply by weights
            total_belief = np.sum(belief)
            if(total_belief>0):
                avg_coords = np.sum(n_arr, axis=0) / np.sum(belief)
            else:
                avg_coords = np.mean(coords,axis=0)
            return avg_coords
        elif(method =='max_belief'):
            ind = np.amax(belief)
            return coords[ind]

    def getAllLandmarkCoordinates(self):
        point_ids = list(self.map_df['LANDMARK'])
        coords_origin = self.getCoordsFromLandmarks(point_ids, 1)
        coords_destiny = self.getCoordsFromLandmarks(point_ids, 2)
        return coords_origin,coords_destiny, point_ids

    def getTypeIndices(self):
        return list(self.map_df["TYPE"]),list(self.map_df["LANDMARK"])

    def getStateIndices(self):
        return list(self.map_df["STATE"]),list(self.map_df["LANDMARK"])
    
    def getTotalCalibration(self):
        """---------------------------------------
            Returns the number of calibrated points.
            
        """
        return len(self.map_df.loc[self.map_df['TYPE'] == PointType.calibrated])

    def deleteCalibrations(self):
        coordsOrigin, coordsDestiny, point_ids = self.getLandmarksByType(PointType.calibrated)
        are_protected = []
        are_blocked = []
        for el in point_ids:
            if self.is_protected(el):
                are_protected.append(el)
            elif self.is_blocked(el):
                are_blocked.append(el)
            self.deleteLandmark(el,False,False)
        self.CalibratedPtp.reset()
        self.GlobalPtp.reset()
        self.addSetPoints(coordsOrigin,[],point_ids,PointType.non_calibrated, are_protected, are_blocked)
        self.updateMap()
        return

    def getTotalLandmarksByType(self,type):
        return len(self.map_df.loc[self.map_df['TYPE'] == type])

####################################################################
    def blockPoint(self,point_id):
        self.changeState(point_id,State.blocked)
        # self.cde_df.loc[self.cde_df['LANDMARK'] == point_id,'BELIEF'] = 0
        # self.cor_df.loc[self.cor_df['LANDMARK'] == point_id, 'BELIEF'] = 0

    def unblockPoint(self,point_id):
        self.changeState(point_id, State.zeroed)

    def changeState(self, point_id, state):
        
        self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'STATE'] = state
        

    def changeType(self,point_id,type, updateModel = False):
        
        self.map_df.loc[self.map_df['LANDMARK'] == point_id,'TYPE'] = type
        if type == PointType.calibrated :
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_ORIGIN'] = False
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_DESTINY'] = False
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_TAG'] = False
        elif type == PointType.target or type == PointType.non_calibrated :
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_ORIGIN'] = False
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_DESTINY'] = True
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_TAG'] = False
        elif type == PointType.acquired:
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_ORIGIN'] = True
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_DESTINY'] = False
            self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'UPDATE_TAG'] = False
        
        if updateModel:
            self.updateMap(point_id)

    def is_blocked(self,point_id):
        return self.checkState(point_id,State.blocked)

    def is_protected(self, point_id):
        return self.checkState(point_id,State.protected)

    def are_protected(self, point_id_list):
        prot_list = []
        for el in point_id_list:
            if self.is_protected(el):
                prot_list.append(el)
        return prot_list

    ##################################################################################################################
    #### PROCEDURES
    #################################################################################################################
    @abstractmethod
    def ready(self):
        pass

    def addCoordsOrigin(self,point_id, coords, belief):
        
        if not (self.isEmpty(coords)):
            df2 = self.cor_df['LANDMARK'] == point_id
            dindex = df2.index[df2 == True].tolist()
            if dindex:
                my_ind = dindex[0]
                self.cor_df.loc[my_ind, self.col_dim_coords_origin] = coords[range(0, len(self.col_dim_coords_origin))]
            else:
                self.cor_df.loc[len(self.cor_df), self.col_dim_coords_origin] = coords[range(0,len(self.col_dim_coords_origin))]
                my_ind = self.cor_df.index[-1]
                self.cor_df.loc[my_ind, 'LANDMARK'] = point_id
            if (belief > 0):
                self.cor_df.loc[my_ind, 'BELIEF'] = belief
            else:
                if (np.isnan(self.cor_df.loc[my_ind, 'BELIEF'])):
                    self.cor_df.loc[my_ind, 'BELIEF'] = 0
                    # otherwise, leave it
        


    def addCoordsDestiny(self,point_id, coords, belief):
        
        if not (self.isEmpty(coords)):
            df2 = self.cde_df['LANDMARK'] == point_id
            dindex = df2.index[df2 == True].tolist()
            if dindex:
                my_ind = dindex[0]
                self.cde_df.loc[my_ind, self.col_dim_coords_destiny] = coords[range(0, len(self.col_dim_coords_destiny))]
            else:
                self.cde_df.loc[len(self.cde_df), self.col_dim_coords_destiny] = coords[range(0,len(self.col_dim_coords_destiny))]
                my_ind = self.cde_df.index[-1]
                self.cde_df.loc[my_ind, 'LANDMARK'] = point_id
            if (belief > 0):
                self.cde_df.loc[my_ind, 'BELIEF'] = belief
            else:
                if (np.isnan(self.cde_df.loc[my_ind, 'BELIEF'])):
                    self.cde_df.loc[my_ind, 'BELIEF'] = 0
                    # otherwise, leave it
        

    def addPoint(self,coords_origin,coords_destiny,point_type,point_id, belief = [1.0,1.0], updateModel=True):
        """
            ACK
        Adds a point to the map.

        We supply:
            coordinates of origin, coordinates of destiny, name
            coordinates of origin, coordinates of destiny, None -> autotag created (or temp_tag waiting to be updated)

            corigin, cdestiny, name  calibrated
            corigin, [ temp], name       non-calibrated, target
            [],cdestiny,name        acquired
            [],[], name        Not accepted
        """


        coords_origin = np.array(coords_origin)
        coords_destiny = np.array(coords_destiny)

        if(coords_destiny.size == 0 and coords_origin.size == 0):
             self.logger.info("No data")
             return -1

        if(not belief):
            belief = [1.0,1.0]

        ## IMPORTANT : THIS SEQUENCE HAS CONDITIONAL DEPENDENCIES
        ## PHASE 1 : CONSISTENCY
        ## ACQUIRED POINTS
        ## DEFINITION : landmarks acquired on the destiny coordinates
        ##              normally they don't have origin coordinates
        ##              name is usually generated
        if(point_type == PointType.acquired):
            # If I donot have destiny coordinates... ERROR
            if (self.isEmpty(coords_destiny)):
                self.logger.info("From "+str(self.map_id)+": Trying to add ACQUIRED point without coordinates of destiny!!")
                return -1
            # I don't have origin coordinates, then I have to generate them
            if(self.isEmpty(coords_origin)):
                coords_origin = self.point_to_Origin(coords_destiny) # Generate origin coordinates
            # If I have no ID, I will have to generate one
            if(not point_id):
                point_id = self.getAutoTag(coords_origin)
        ## NON-CALIBRATED OR TARGET
        elif (point_type == PointType.non_calibrated or point_type == PointType.target):
            # If I donot have origin coordinates... ERROR
            if (self.isEmpty(coords_origin)):
                self.logger.info("From " + str(self.map_id) + ": Trying to add NON_CAL or TARGET point without coordinates of origin!!")
                return -1
            # I don't have destiny coordinates, then I have to generate them
            if (self.isEmpty(coords_destiny)):
                coords_destiny = self.point_to_Destiny(coords_origin)  # Generate origin coordinates
            # If I have no ID, I will have to generate one
            if (not point_id):
                point_id = self.getAutoTag(coords_destiny)
        # CALIBRATED
        elif(point_type == PointType.calibrated):
            if (self.isEmpty(coords_destiny)):
                self.logger.info("From " + str(self.map_id) + ": Trying to add CALIBRATION point without coordinates of origin AND destiny!!")
                return -1
            if (not point_id):
                if(self.isEmpty(coords_origin)):
                    return -1
                point_id = self.getAutoTag(coords_destiny)
            else:
                if(self.isEmpty(coords_origin)):
                    coords_origin = self.getLandmark(point_id,1)


        ########## Now we check if the landmark is here already #######################
        ## Are we repeating the landmark ?
        if(self.isin(point_id)):
            # Is this a type change?
            ### SAME TYPE
            if(self.checkType(point_id,point_type)):  # IF has the same type, we have to add it (it is a REAL MEASURE of the point)
                self.addCoordsOrigin(point_id,coords_origin,belief[0])
                self.addCoordsDestiny(point_id,coords_destiny,belief[1])
                if(updateModel):
                    self.updateMap(point_id)
                self.last_point_added = point_id
                return point_id
            else:
                ## PHASE 2 : We have to check types and upgrade them accordingly
                ### non_calibrated -> calibrated
                ### This code is henious. Has to be refactored
                ###
                old_type = self.map_df.loc[self.map_df['LANDMARK'] == point_id, 'TYPE']
                old_type = old_type.iloc[0]
                self._changeTypeAndAdd(old_type,point_type,point_id,coords_origin,coords_destiny, belief, updateModel)
        else: # NORMAL new acquisition

            my_ind = len(self.map_df)
            
            self.map_df.loc[my_ind, :] = 0
            self.map_df.loc[my_ind, "LANDMARK"] = point_id
            
            self.addCoordsOrigin(point_id, coords_origin, belief[0])
            self.addCoordsDestiny(point_id, coords_destiny, belief[1])

            
            if (point_type == PointType.calibrated):
                self.map_df.loc[my_ind, "TYPE"] = PointType.calibrated
                self.map_df.loc[my_ind, "UPDATE_ORIGIN"] = False
                self.map_df.loc[my_ind, "UPDATE_DESTINY"] = False
                self.map_df.loc[my_ind, "UPDATE_TAG"] = False
            elif(point_type==PointType.non_calibrated):
                self.map_df.loc[my_ind, "TYPE"] = PointType.non_calibrated
                self.map_df.loc[my_ind, "UPDATE_ORIGIN"] = False
                self.map_df.loc[my_ind, "UPDATE_DESTINY"] = True

                if ("NO_ID" in point_id):
                    self.map_df.loc[my_ind, "UPDATE_TAG"] = True
                else:
                    self.map_df.loc[my_ind, "UPDATE_TAG"] = False
            elif(point_type==PointType.target):
                self.map_df.loc[my_ind, "TYPE"] = PointType.target
                self.map_df.loc[my_ind, "UPDATE_ORIGIN"] = False
                self.map_df.loc[my_ind, "UPDATE_DESTINY"] = True

                if ("NO_ID" in point_id):
                    self.map_df.loc[my_ind, "UPDATE_TAG"] = True
                else:
                    self.map_df.loc[my_ind, "UPDATE_TAG"] = False
            elif (point_type == PointType.acquired):
                self.map_df.loc[my_ind, "TYPE"] = PointType.acquired
                self.map_df.loc[my_ind, "UPDATE_ORIGIN"] = True
                self.map_df.loc[my_ind, "UPDATE_DESTINY"] = False

                if ("NO_ID" in point_id):
                    self.map_df.loc[my_ind, "UPDATE_TAG"] = True
                else:
                    self.map_df.loc[my_ind, "UPDATE_TAG"] = False
            else:
                self.logger.error("From " + str(self.map_id) + ":ERROR, type of point not found.")
                return
            if (self.isEmpty(coords_origin)):  # Not enough info to generate them
                self.map_df.loc[my_ind, "UPDATE_ORIGIN"] = True
            if (self.isEmpty(coords_destiny)):  # Not enough info to generate them
                self.map_df.loc[my_ind, "UPDATE_DESTINY"] = True

            self.list_errorOrigin[point_id] = []
            self.list_errorDestiny[point_id] = []

            self.map_df.loc[my_ind, "RMS_AVG"] = 0
            self.map_df.loc[my_ind, "RMS_SD"] = 0

            self.list_local_area[point_id] = LocalArea()
            
            self.updateLandmark(point_id)

        self.last_point_added = point_id

        return point_id

    def _changeTypeAndAdd(self,old_type, point_type, point_id, coords_origin, coords_destiny, belief, updateModel = False):
        """
        First refactoring step.

        :param old_type:
        :param point_type:
        :param point_id:
        :param coords_origin:
        :param coords_destiny:
        :param belief:
        :param updateModel:
        :return:
        """
        if (old_type == PointType.calibrated and point_type == PointType.non_calibrated):  # Update coordinates origin, that's all
            self.addCoordsOrigin(point_id, coords_origin, belief[0])
        else:
            if (old_type == PointType.non_calibrated and point_type == PointType.calibrated):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.addCoordsDestiny(point_id, coords_destiny, belief[1])
                self.changeType(point_id, PointType.calibrated, updateModel)

            elif (old_type == PointType.calibrated) and (point_type == PointType.non_calibrated):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.addCoordsDestiny(point_id, coords_destiny, belief[1])
                self.changeType(point_id, PointType.non_calibrated)

            elif (old_type == PointType.acquired) and (point_type == PointType.non_calibrated):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.changeType(point_id, PointType.calibrated, updateModel)

            elif (old_type == PointType.non_calibrated) and (point_type == PointType.acquired):
                self.addCoordsDestiny(point_id, coords_destiny, belief[1])
                self.changeType(point_id, PointType.calibrated, updateModel)

            elif (old_type == PointType.target) and (point_type == PointType.non_calibrated):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.changeType(point_id, PointType.non_calibrated)

            elif (old_type == PointType.non_calibrated) and (point_type == PointType.target):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.changeType(point_id, PointType.target)

            elif (old_type == PointType.calibrated) and (point_type == PointType.target):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.changeType(point_id, PointType.target)

            elif (old_type == PointType.acquired) and (point_type == PointType.target):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.changeType(point_id, PointType.target)

            elif (old_type == PointType.target) and (point_type == PointType.acquired):
                self.addCoordsDestiny(point_id, coords_destiny, belief[1])
                self.changeType(point_id, PointType.target)

            elif (old_type == PointType.calibrated) and (point_type == PointType.acquired):
                self.addCoordsDestiny(point_id, coords_destiny, belief[1])
                self.changeType(point_id, PointType.calibrated, updateModel)

            elif (old_type == PointType.target) and (point_type == PointType.calibrated):
                self.addCoordsOrigin(point_id, coords_origin, belief[0])
                self.changeType(point_id, PointType.target)
            else:
                self.logger.info(" Old type :" + str(old_type) + " New type :" + str(point_type))
                self.logger.info("From " + str(self.map_id) + ": Change of type not supported for " + point_id + ".")
                return -1
            return


    def addSetPoints(self, pointsOrigin, pointsDestiny, names, point_type, protected_list = None, blocked_list = None, update_origin=False, update_destiny=True, updateModel=True):

        pointsDestiny = np.array(pointsDestiny, dtype=np.float32)
        pointsOrigin = np.array(pointsOrigin, dtype=np.float32)

        if (not bool(pointsDestiny.shape) or pointsDestiny.size == 0):
            s = (pointsOrigin.shape)
            pointsDestiny = np.empty(s)
            pointsDestiny.fill(np.inf)

        if (not bool(pointsOrigin.shape) or pointsOrigin.size == 0):
            s = (pointsDestiny.shape)
            pointsOrigin = np.empty(s)
            pointsOrigin.fill(np.inf)

        if len(pointsOrigin.shape)<1:  # Failsafe, should be changed by exception
            return
        s,d = pointsOrigin.shape
        for i in range(s):
            coords_lm = pointsOrigin[i]
            coords_sem = pointsDestiny[i]
            nid = names[i]
            self.addPoint(coords_lm, coords_sem, point_type, nid, [1.,1.], False)
            if nid in protected_list:
                self.changeState(nid,State.protected)
            if nid in blocked_list:
                self.changeState(nid,State.blocked)
            self.updateOrigin(nid,update_origin)
            self.updateDestiny(nid,update_destiny)

        if(updateModel):
            self.updateMap()

    def updateOrigin(self,point_id, bool_up):
        self.map_df.loc[self.map_df["LANDMARK"]==point_id, "UPDATE_ORIGIN"] = bool_up

    def updateDestiny(self, point_id, bool_up):
        self.map_df.loc[self.map_df["LANDMARK"] == point_id, "UPDATE_DESTINY"] = bool_up

    def replaceLandmark(self, coords_origin, coords_destiny, belief, point_id):
        """
        Keeps the identity of the point (type), but updates values, erasing EVERYTHING
        :param coords_origin:
        :param coords_destiny:
        :param point_id:
        :return:
        """
        if (not self.isin(point_id)):
            return -1
        if (not self.isEmpty(coords_origin)):
            self.cor_df = self.cor_df[self.cor_df.LANDMARK != point_id]
            self.addCoordsOrigin(point_id,coords_origin,belief[0])
            self.list_errorOrigin[point_id] = []
        if (not self.isEmpty(coords_destiny)):
            self.cde_df = self.cde_df[self.cde_df.LANDMARK != point_id]
            self.addCoordsDestiny(point_id, coords_destiny,belief[1])
            self.list_errorDestiny[point_id] = []
        #self.map_df.loc[self.map_df['LANDMARK'] == point_id, self.col_reset] = 0
        self.list_local_area[point_id] = LocalArea()

    def updateLastLandmark(self, coords_origin, coords_destiny, point_id, protect = False, which_up = 0):
        if (not self.isin(point_id)):
            return False

        if which_up == 0 :
            up_or = True
            up_dest = True
        elif which_up == 1:
            up_or = True
            up_dest = False
        elif which_up == 2 :
            up_or = False
            up_dest = True
        else:
            return False

        df2 = self.cde_df['LANDMARK'] == point_id
        dindex = df2.index[df2 == True].tolist()
        if not dindex:
            self.logger.info("Coordinates not found for "+point_id+". Give coordinates of reference first before securing.")
            return False
        else:
            dindex = dindex[0]

        if np.any(np.array(coords_origin).shape) and up_or :
            distance = virtualGridMap.dist(np.array([self.cor_df.COORDS_DESTINY_X[dindex], self.cor_df.COORDS_DESTINY_Y[dindex]]),
                                           np.array([coords_origin[0], coords_origin[1]]))
            self.logger.info( "From " + str(self.map_id) + ":Point " + point_id + " corrected. Difference is:" + str(distance))
            self.addCoordsOrigin(point_id, coords_origin, 1.0)


        if np.any(np.array(coords_destiny).shape) and up_dest:
            distance = virtualGridMap.dist(
                np.array([self.cde_df.COORDS_DESTINY_X[dindex], self.cde_df.COORDS_DESTINY_Y[dindex]]),
                np.array([coords_destiny[0], coords_destiny[1]]))
            self.logger.info(
                "From " + str(self.map_id) + ":Point " + point_id + " corrected. Difference is:" + str(distance))
            self.addCoordsDestiny(point_id, coords_destiny, 1.0)


        self.list_local_area[point_id] = LocalArea()
        if protect :
            
            self.changeState(point_id,State.protected)
            self.map_df.loc[dindex, "UPDATE_ORIGIN"] = False
            self.map_df.loc[dindex, "UPDATE_DESTINY"] = False
            self.map_df.loc[dindex,"UPDATE_TAG"] = False
            

        self.updateLandmark(point_id)
        return True

    def getMeasuresPoint(self,point_id,map_value, removeInfs = True):
        if(not self.isin(point_id)):
           return []

        if (map_value == 1):
            data_df = self.cor_df.loc[self.cor_df['LANDMARK'] == point_id]
            if(removeInfs):
                data_df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.col_dim_coords_origin, how = "all")
                return np.array(data_df[self.col_dim_coords_origin])
        elif (map_value == 2):
            data_df = self.cde_df.loc[self.cde_df['LANDMARK'] == point_id]
            if (removeInfs):
                data_df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.col_dim_coords_destiny, how="all")
            return np.array(data_df[self.col_dim_coords_destiny])
        else:
            self.logger.error("From " + str(self.map_id) + ":ERROR: Use map_value 1 to origin, 2 to destiny.")
            return []

    def setMeasuresPoint(self,point_id,map_value,measures, beliefs =[] ):
        

        if (not self.isin(point_id)):
            self.logger.error("From " + str(self.map_id) + ":ERROR: you need to provide a name for the point. Not valid:" + str(point_id))
            return

        if (map_value == 1):
            self.cor_df.loc[self.cor_df['LANDMARK'] == point_id, self.col_dim_coords_origin] = measures[range(0,len(self.col_dim_coords_origin))]
            if (not beliefs):
                self.cor_df.loc[self.cor_df['LANDMARK'] == point_id, "BELIEF"] = 1.0
            else:
                self.cor_df.loc[self.cor_df['LANDMARK'] == point_id, "BELIEF"] = beliefs
        elif (map_value == 2):
            self.cde_df.loc[self.cde_df['LANDMARK'] == point_id,self.col_dim_coords_destiny] = measures[range(0,len(self.col_dim_coords_destiny))]
            if(not beliefs):
                self.cde_df.loc[self.cde_df['LANDMARK'] == point_id, "BELIEF"] = 1.0
            else:
                self.cde_df.loc[self.cde_df['LANDMARK'] == point_id, "BELIEF"] = beliefs
        else:
            self.logger.error("From " + str(self.map_id) + ":ERROR: Use map_value 1 to origin, 2 to destiny.")

        
        return

    def deleteLandmark(self,point_id, updateMap = True, verbose = True):
        """

            Deletes a point to the map. Please, call update map if you use this function
            vmap.updateMap()

        """
        # Do we have a name for the point?
        if(not self.isin(point_id)):
            self.logger.error("From "+str(self.map_id)+":ERROR: you need to provide a name for the point. Not valid:"+str(point_id))
            return
        

        self.map_df = self.map_df[self.map_df.LANDMARK != point_id]
        if not self.cor_df.LANDMARK.empty:
            self.cor_df = self.cor_df[self.cor_df.LANDMARK != point_id]
        if not self.cde_df.LANDMARK.empty:
            self.cde_df = self.cde_df[self.cde_df.LANDMARK != point_id]

        self.list_errorOrigin.pop(point_id,0)
        self.list_errorDestiny.pop(point_id,0)
        self.list_local_area.pop(point_id,0)

        ##    
        if(self.last_point_added == point_id and len(self.map_df)>0):
            self.last_point_added = self.map_df.LANDMARK[0]

        self.map_df = self.map_df.reset_index(drop=True)
        self.cor_df = self.cor_df.reset_index(drop=True)
        self.cde_df = self.cde_df.reset_index(drop=True)

        
        if updateMap:
            self.updateMap(point_id)

        if verbose:
            self.logger.info("From "+str(self.map_id)+":Deleted point :"+point_id)
        return point_id


    def updateTransformPoint(self, pointid):
        ptype = self.getLandmarkType(pointid)
        # I have to update closest neighbors
        coords_ref = self.getLandmark(pointid,1)
        neightags = self.GlobalPtp.getNeighs(coords_ref, k=20)  # Update closest 20 neighbors

        for neigh in neightags :
            if ptype == PointType.acquired:
                coords_p = self.getLandmark(pointid, 2)
            elif ptype in [PointType.non_calibrated, PointType.target, PointType.unknown]:
                coords_p = self.getLandmark(pointid, 1)
            else:
                continue
            self.CalibratedPtp.updateLocalArea(neigh, coords_p)
        return

    def updateTransform(self):
        """
            ACK
            Transformation  between canvas points and the inverse must be stored in the canvas
            It also calculates H_inv, given a coordinate from the destiny  (like user click in canvas).
            At 1 point, nothing is know, the map relies on prior information.
            With 3 or more points, Least squares can be used. We have to avoid the risk of points being colinear.
            With 6 or more, homography with ransac works better.
            With 10 or more, local homography is used for the 6 closest points or more given a specified radius.
            This is uses the ptp (point to point) protocol, taking closest points to the region selected and
            calculating the closest in the region.
        """

        coordCalibrationsOrigin, coordCalibrationsDestiny, namesCalibration  = self.getLandmarksByType(PointType.calibrated)

        total_calibration_points = len(coordCalibrationsOrigin)
        if (total_calibration_points < 2):
            self.logger.info("From " + str(self.map_id) + ":Not enough points to update.")
            return

        self.logger.info("From "+str(self.map_id)+":Updating transform with " + str(total_calibration_points) + " reference points")
        origin = np.zeros((total_calibration_points, 2), dtype=np.float32)
        destiny = np.zeros((total_calibration_points, 2), dtype=np.float32)
        for i in range(total_calibration_points):
            origin[i, 0] = coordCalibrationsOrigin[i][0]
            origin[i, 1] = coordCalibrationsOrigin[i][1]
            destiny[i, 0] = coordCalibrationsDestiny[i][0]
            destiny[i, 1] = coordCalibrationsDestiny[i][1]


        self.CalibratedPtp.updateGlobal(origin,destiny,namesCalibration)

        coordT, _, namesTarget = self.getLandmarksByType(PointType.target)
        self.processLocalArea(coordT,namesTarget)

        coordNC, _, namesNonCal = self.getLandmarksByType(PointType.non_calibrated)
        self.processLocalArea(coordNC, namesNonCal)

        _, coordACQ, namesAcq = self.getLandmarksByType(PointType.acquired)
        self.processLocalArea(coordACQ, namesAcq)

        namesAll = self.getLandmarkIds()
        originAll = self.getCoordsFromLandmarks(namesAll,1)
        destinyAll = self.getCoordsFromLandmarks(namesAll,2)
        self.GlobalPtp.updateGlobal(originAll, destinyAll, namesAll)


    def processLocalArea(self, coords_p, namePoints):
        if (len(namePoints) > 0):  # if there are targets, we go target centric
            local_areas = [(self.list_local_area[nameT], coords_p[i]) for i, nameT in enumerate(namePoints)]
            with closing(ThreadPool(8)) as pool:
                pool.starmap(self.CalibratedPtp.updateLocalArea, local_areas)



    def getRMSError(self):
        if(self.rms_avg):
            return (self.rms_avg[-1],self.rms_sd[-1])
        else:
            return ("NA","NA")
     
    def getErrorList(self,point_id):
        error_retro = [np.inf]
        if point_id in self.list_local_area.keys():
            error_retro = self.list_local_area[point_id].getErrorRetro()
            
        return error_retro

    @abstractmethod    
    def point_to_Origin(self,coord_destiny, point_id=""):
        pass
    
    @abstractmethod  
    def point_to_Destiny(self,coord_origin,point_id=""):
        pass
    
    @abstractmethod  
    def getAutoTag(self, coord_):
        pass

    def getRadius(self,p_id,map_value):

        if(map_value == 1):
                return self.list_local_area[p_id].radius_origin
        elif(map_value == 2):
                return self.list_local_area[p_id].radius_destiny
        else:
            return

    def setRadius(self,p_id,radius,map_value):
        list_ids = self.getLandmarkIds()
        if p_id not in list_ids:
            return 0
        #######################################
        if(map_value == 1):
                self.list_local_area[p_id].radius_origin = radius
                self.list_local_area[p_id].radius_destiny = self.conversion_ratio*radius
        elif(map_value == 2):
                self.list_local_area[p_id].radius_origin = radius*(1/self.conversion_ratio)
                self.list_local_area[p_id].radius_destiny = radius
        else:
            return 0
        self.CalibratedPtp.updateLocalArea(self.list_local_area[p_id])
        return self.conversion_ratio*radius

    def getRadii(self,map_value):
        list_radius = []
        if(map_value == 1):
            for el in self.list_local_area.values():
                list_radius.append(el.radius_origin)
            return np.array(list_radius)
        else:
            for el in self.list_local_area.values():
                list_radius.append(el.radius_destiny)
            return np.array(list_radius)

    def applyTransform(self,M, map_value):
        coordOrigin, coordDestiny, pointids = self.getAllLandmarkCoordinates()

        if map_value == 1 :
            homo = (np.asmatrix(coordOrigin) + np.array([0., 0., 1.]))
            res = np.dot(M, homo.T)
            aux_df = pd.DataFrame(columns=self.columns_corigin)
            aux_df['LANDMARK'] = pointids
            aux_df.loc[:, 'COORDS_ORIGIN_X'] = np.squeeze(np.asarray(res[0]))
            aux_df.loc[:, 'COORDS_ORIGIN_Y'] = np.squeeze(np.asarray(res[1]))
            aux_df.loc[:, 'COORDS_ORIGIN_Z'] = 0.0
            aux_df.loc[:, 'BELIEF'] = 0.0
            self.cor_df = aux_df
        elif map_value == 2 :
            homo = (np.asmatrix(coordDestiny) + np.array([0., 0., 1.]))
            res = np.dot(M, homo.T)
            aux_df = pd.DataFrame(columns=self.columns_cdestiny)
            aux_df['LANDMARK'] = pointids
            aux_df.loc[:, 'COORDS_DESTINY_X'] = np.squeeze(np.asarray(res[0]))
            aux_df.loc[:, 'COORDS_DESTINY_Y'] = np.squeeze(np.asarray(res[1]))
            aux_df.loc[:, 'COORDS_DESTINY_Z'] = 0.0
            aux_df.loc[:, 'BELIEF'] = 0.0
            self.cde_df = aux_df
        else:
            raise ValueError
        self.updateMap()

    @abstractmethod
    def applyTransformToPoint(self,M, coords):
        """
        Maps a point from one coordinates to others using M

        """
        pass

    def checkCoordinateLimits(self,coord):
        """
        :param coord:
        :return:
        """
        raise NotImplementedError()

    def getAutoTag(self, coords):
        if (coords[0] == -1 or np.isinf(coords[0])):
            tag = random.randint(1, 100000)  # Integer from 1 to 10, endpoints included
            point_id = "NO_ID_" + str(tag)
            list_ids = self.getLandmarkIds()
            if point_id in list_ids:
                rval = random.randint(2, 10) * 100000
                tag = random.randint(100000, rval)
                point_id = "NO_ID_" + str(tag)
            return point_id
        cx = round(coords[0])
        cy = round(coords[1])
        point_id = self.grid_map.getTag(cx, cy)
        if(self.isin(point_id)):
            # Has numbers?
            if ("_n_" in point_id):
                indx = point_id.find("_n_")
                seq = point_id[indx + 3:]
                point_id = point_id[:indx + 3] + str(int(seq) + 1)
            else:
                point_id = point_id + "_n_1"
        return point_id

    def updateMap(self, pointid = None):

        if pointid:
            self.updateLandmark(pointid)
            self.updateTransformPoint(pointid)
        else:
            self.updateLandmarks()
            self.updateTransform()

        update = False
        # for each point that coordinates must be updated

        if(np.any(self.CalibratedPtp.Hg_inv)):
            df_to_up = self.map_df.loc[self.map_df["UPDATE_DESTINY"]]
            listids = df_to_up["LANDMARK"]
            for el in listids:
                coord_origin = df_to_up.loc[df_to_up["LANDMARK"]==el,self.col_dim_coords_origin]
                n_coord = self.point_to_Destiny(np.squeeze(coord_origin.values),el)
                self.addCoordsDestiny(el, n_coord, 0)
                if(np.any(df_to_up.loc[df_to_up["LANDMARK"] == el, "UPDATE_TAG"])):
                    self.map_df.loc[self.map_df["LANDMARK"] == el, "UPDATE_TAG"] = False
                    tag = self.getAutoTag(n_coord)
                    
                    self.map_df.loc[self.map_df["LANDMARK"] == el,'LANDMARK'] = tag
                    self.cor_df.loc[self.cor_df["LANDMARK"] == el,'LANDMARK'] = tag
                    self.cde_df.loc[self.cde_df["LANDMARK"] == el,'LANDMARK'] = tag
                    
            update = True

        if(np.any(self.CalibratedPtp.Hg)):
            df_to_up = self.map_df.loc[self.map_df["UPDATE_ORIGIN"]]
            listids = df_to_up["LANDMARK"]
            for el in listids:
                coord_destiny = df_to_up.loc[df_to_up["LANDMARK"] == el, self.col_dim_coords_destiny]
                n_coord = self.point_to_Origin(np.squeeze(coord_destiny.values),el)
                self.addCoordsOrigin(el,n_coord, 0)
                if(np.any(df_to_up.loc[df_to_up["LANDMARK"] == el, "UPDATE_TAG"])):
                    self.map_df.loc[self.map_df["LANDMARK"] == el, "UPDATE_TAG"] = False
                    tag = self.getAutoTag(n_coord)
                    
                    self.map_df.loc[self.map_df["LANDMARK"] == el,'LANDMARK'] = tag
                    self.cor_df.loc[self.cor_df["LANDMARK"] == el,'LANDMARK'] = tag
                    self.cde_df.loc[self.cde_df["LANDMARK"] == el,'LANDMARK'] = tag
                    
            update = True

        if(update):
            self.updateLandmarks()



    def to_dict(self):
        self.updateMap()
        map_dict = self.map_df.to_json()
        cor_dict = self.cor_df.to_json()
        cde_dict = self.cde_df.to_json()

        f_dict = {}
        f_dict["MAP"] = map_dict
        f_dict["COR"] = cor_dict
        f_dict["CDE"] = cde_dict
        f_dict["MAP_ID"] = self.map_id
        return f_dict

    def getLocalArea(self,pid):
        return self.list_local_area[pid]

    def getNN(self,coords,map_id,k, types = None):
        if(not types):
            return self.CalibratedPtp.getNeighs(coords,0,k, map_id)
        else:
            # We have to get all data based on types
            all_data = []
            all_ids = []
            for el in types:
                coordOrigin, coordDestiny, ids = self.getLandmarksByType(el)
                if(map_id==1):
                    m_coords = coordOrigin[:,0:len(coords)]
                else:
                    m_coords = coordDestiny[:,0:len(coords)]
                if(np.any(np.isinf(m_coords)) or np.any(np.isnan(m_coords))):
                    continue
                else:
                    all_data.append(m_coords)
                    all_ids = all_ids + ids


            all_data = np.vstack(all_data)
            if(not np.any(all_data)):
                return [-np.inf,-np.inf]
            tree = spt.KDTree(all_data)
            dists,inds = tree.query(coords, k)
            # If there is no more points it returns inf, and this has to be removed
            to_del = np.where(np.isinf(dists))
            dists = np.delete(dists,to_del)
            inds = np.delete(inds,to_del)
            all_ids = np.array(all_ids)
            return  all_data[inds],list(all_ids[inds]),dists

############################################################################################
#
#   LM
#
#############################################################################################        
class virtualGridMapLM(virtualGridMap):
    """
    In LM, ORIGIN will be the Canvas
           DESTINY will be the LM
           i.e. map Canvas_LM
    LM and Canvas maps adjust to the specific grid pattern we provide, in our case
    a Mattek grid dish.
    It is a 2D Map
    """   
       
    def __init__(self,logger):
        super(virtualGridMapLM, self).__init__(logger,force2D=True)

    def ready(self):
        return (len(self.getLandmarkIDsByType(PointType.calibrated))>2)

    def applyTransformToPoint(self, M, coords_origin):
        """
        Maps a point from the origin to the destiny origin of coordinates
        The transformation matrix M is provided and only works for 2D points

        """
        x = coords_origin[0]
        y = coords_origin[1]
        if M is not None:
            tmp = np.float32([x, y, 1.0])
            trh = np.dot(M, tmp.transpose())  # trh = self.H.Transform(tmp.transpose())
            # a = np.array([point],dtype='float32')
            # a = np.array([a])
            # pointOut= cv2.perspectiveTransform(a,ret)
            trh /= trh[2]
            # ERROR checking
            if (trh[0] < 0 or trh[1] < 0):
                self.logger.warning("[", trh[0], ",", "]")
                self.logger.warning("From "+str(self.map_id)+":ERROR: negative coordinates")
                if (trh[0] < 0):
                    trh[0] = 0.0
                if (trh[1] < 0):
                    trh[1] = 0.0
            return (trh[0:2])
        else:
            return (np.array([-1, -1]))

    def point_to_Destiny(self, coords_origin,point_id =""):
        """
            Maps a point from the origin to the destiny origin of coordinates
        
        """

        x = coords_origin[0]
        y = coords_origin[1]
        if self.CalibratedPtp.Hg is not None:
            tmp = np.float32([x,y,1.0])
            neighs,_ = self.CalibratedPtp._getNeighs([x,y],map_id=1)
            Hl,_ = self.CalibratedPtp.getLocalTransform(neighs)
            trh = np.dot(Hl, tmp.transpose())  #  trh = self.H.Transform(tmp.transpose())
            # a = np.array([point],dtype='float32')
            # a = np.array([a])
            # pointOut= cv2.perspectiveTransform(a,ret)
            trh /= trh[2]
            return(trh[0:2])      
        else:
            return(np.array([-np.inf,-np.inf]))

    def point_to_Origin(self,coords_destiny,point_id =""):
        """
            Maps a point from the destiny to the origin of coordinates
        
        """
        x = coords_destiny[0]
        y = coords_destiny[1]
        if self.CalibratedPtp.Hg_inv is not None:
            tmp = np.float32([x,y,1.0])
            neighs,_ = self.CalibratedPtp._getNeighs([x,y],map_id = 2)
            Hlinv,_ = self.CalibratedPtp.getLocalTransform(neighs,inverse=True)
            trh = np.dot(Hlinv, tmp.transpose()) #  trh = self.H.Transform(tmp.transpose())
            # a = np.array([point],dtype='float32')
            # a = np.array([a])
            # pointOut= cv2.perspectiveTransform(a,ret)
            trh /= trh[2]
            return(trh[0:2])     
        else:
            return(np.array([-1,-1]))

    def getAutoTag(self,coords_destiny):
        if(coords_destiny[0]==-1):
            tag = random.randint(1, 100000)  # Integer from 1 to 10, endpoints included
            point_id = "NO_ID_"+str(tag)
            if(self.isin(point_id)):
                rval = random.randint(2,10)*100000
                tag = random.randint(100000,rval)
                point_id = "NO_ID_"+str(tag)
            return point_id
        cx = round(coords_destiny[0])
        cy = round(coords_destiny[1])
        point_id = self.grid_map.getTag(cx,cy)
        if(self.isin(point_id)):
            # Has numbers?
            if("_n_" in point_id):
                indx = point_id.find("_n_")
                seq  = point_id[indx+3:]
                point_id  = point_id[:indx+3]+"_%08d"%(int(seq) + 1,0)
            else:
                point_id = point_id+"_n_1"
        return point_id


    ###############################################################################


    ######################### MATTEK DISH specific #################################333
    def find_closest_letter(self, coords, map_value):
        """
        Given a set of 2D coordinates in the plane, it gives you back the closest landmark
        0 Canvas 1 LM
        Assuming Canvas_LM Map
        """
        return self.grid_map.find_closest_letter(coords,map_value)

    def find_square_letter(self, coords, map_value, do_round=False):
        """
        Given a set of 2D coordinates in the plane, it gives you back the closest landmark
        0 Canvas 1 LM
        Assuming Canvas_LM Map
        """
        ncoords = []
        letter = ''
        if (map_value == 1):
            ncoords = self.point_to_Origin(coords)
            if(do_round == True):
                ncoords = np.round(ncoords)
        else:
            ncoords = coords
        return self.grid_map.find_square_letter(ncoords)

    # def createVirtualGraph(self,tags,datalm,datamap,points_list,orientation):
    #    self.grid_map.populateGraph(tags,datalm,datamap,points_list,orientation)
    #    self.addSetPoints(self.grid_map.map_coordinates_origin.values(),self.grid_map.map_coordinates_destiny.values(),self.grid_map.map_coordinates_origin.keys(),'CALIBRATED')
    def getLetterCoordinates(self, letter, map):
       coords = []
       try:
            if(map == 0):
                return self.grid_map.map_coordinates_origin[letter]
            else:
                return self.grid_map.map_coordinates_destiny[letter]
       except KeyError:
            print('Letter '+str(letter)+ 'does not exist.' )
            return ''

########################################################################################
#       
#     SEM
#       
#####################################################################################

class virtualGridMapSEM(virtualGridMap):


    def __init__(self,logger):
        super(virtualGridMapSEM, self).__init__(logger)
        self.map = Map(scale=1e-3)
        self.zmap = ZMap()

    def ready(self):
        return (len(self.getLandmarkIDsByType(PointType.calibrated)) > 2)

    def applyTransformToPoint(self, M, coords_origin):
        """
        Maps a point from the origin to the destiny origin of coordinates

        """
        x = coords_origin[0]
        y = coords_origin[1]
        if M is not None:
            tmp = np.float32([x, y, 1.0])
            trh = np.dot(M, tmp.transpose())  # trh = self.H.Transform(tmp.transpose())
            # a = np.array([point],dtype='float32')
            # a = np.array([a])
            # pointOut= cv2.perspectiveTransform(a,ret)
            if(M.shape[0]==3): # Perspective Transform, but not affine...
                trh /= trh[2]
            # ERROR checking
            if (trh[0] < 0 or trh[1] < 0):
                self.logger.info("[", trh[0], ",", "]")
                self.logger.info("From "+str(self.map_id)+":ERROR: negative coordinates")
                if (trh[0] < 0):
                    trh[0] = 0.0
                if (trh[1] < 0):
                    trh[1] = 0.0
            return np.array([trh[0], trh[1], 0.0])
        else:
            return (np.array([-1, -1, -1]))

    def point_to_Destiny(self, coords_origin, point_id=""):
        """
            Maps a point from the origin to the destiny origin of coordinates

        """

        x = coords_origin[0]
        y = coords_origin[1]
        if self.CalibratedPtp.Hg is not None:
            tmp = np.float32([x, y, 1.0])
            if (self.isin(point_id)):
                H = self.list_local_area[point_id].getTransform()
                if H is not None:
                    trh = np.dot(H, tmp.transpose())
                else:
                    trh = np.dot(self.CalibratedPtp.Hg, tmp.transpose())
            else:
                trh = np.dot(self.CalibratedPtp.Hg, tmp.transpose())  # trh = self.H.Transform(tmp.transpose())
            # a = np.array([point],dtype='float32')
            # a = np.array([a])
            # pointOut= cv2.perspectiveTransform(a,ret)
            trh /= trh[2]
            return np.array([trh[0],trh[1],0.0])
        else:
            return (np.array([-np.inf, -np.inf, -np.inf]))

    def point_to_Origin(self, coords_destiny, point_id=""):
        """
            Maps a point from the destiny to the origin of coordinates

        """
        x = coords_destiny[0]
        y = coords_destiny[1]
        if self.CalibratedPtp.Hg_inv is not None:
            tmp = np.float32([x, y, 1.0])
            if (self.isin(point_id)):
                Hinv = self.list_local_area[point_id].getTransform(inverse=True)
                if Hinv is not None:
                    trh = np.dot(Hinv, tmp.transpose())
                else:
                    trh = np.dot(self.CalibratedPtp.Hg_inv, tmp.transpose())
            else:
                trh = np.dot(self.CalibratedPtp.Hg_inv, tmp.transpose())  # trh = self.H.Transform(tmp.transpose())
            # a = np.array([point],dtype='float32')
            # a = np.array([a])
            # pointOut= cv2.perspectiveTransform(a,ret)
            trh /= trh[2]
            return np.array([trh[0],trh[1],0.0])
        else:
            return (np.array([-np.inf, -np.inf, -np.inf]))


   ###############################################################################

    ############ Mattek specific
    def find_closest_letter(self, coords, map_value):
        """
        Given a set of 2D coordinates in the plane, it gives you back the closest landmark

        Assuming Canvas_LM Map
        """
        ncoords = []
        letter = ''
        if (map_value == 1):
            ncoords = self.point_to_Origin(coords)
        else:
            ncoords = coords
        return self.grid_map.find_closest_letter(ncoords);

    def getLetterCoordinates(self, letter_id, map_value):
        letter = ''
        coords = self.grid_map.map_labels[letter_id]
        if (map_value == 1):
            return coords
        else:
            if(self.isin(letter_id)):
                return self.getLandmark(letter_id,2)
            else:
                return self.point_to_Destiny(coords)

    def getCornerNeighs(self, letter):

        ctx = self.grid_map.xlabels.index(letter[0])
        cty = self.grid_map.ylabels.index(letter[1])

        n1 = self.grid_map.xlabels[ctx + 1] + self.grid_map.ylabels[cty]
        n2 = self.grid_map.xlabels[ctx + 1] + self.grid_map.ylabels[cty + 1]
        n3 = self.grid_map.xlabels[ctx] + self.grid_map.ylabels[cty + 1]

        return [letter, n1, n2, n3]

    def addToZMap(self,coords_stage,iz_value):
        self.zmap.position.append(coords_stage)
        self.zmap.z_value.append(iz_value)