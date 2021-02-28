# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:48:36 2015

@author: JMS
"""

from virtualGridMap import *
from abc import ABCMeta, abstractmethod
from scipy.spatial import distance
import ast
from skimage import transform as tf
from scipy.spatial import KDTree,distance
from collections import OrderedDict
import random
import sys


class NonValidMap(Exception):
    pass

class VirtualMapManager(object):
    """
        It manages the map, and serves as a explicit interface between microscope and map:
        - The map only takes care of mathematical transformation and carrying the data
          between homologue graphs.
        - The manager uses that information, takes the microscope information and 
          works with it.
        Attributes:
            vMap: A virtualGridMap representing the maps of the sample.
            msc: must come from the main application and it is the microscope control.
            
       NOTE : INITIALIZE A MAP DOES NOT CREATE A MAP     
    """
    msc_server = None
    
    
    def __init__(self, logger, server_msc):
        """
            Stores microscope server
            
        """
        self.msc_server = server_msc
        self.logger = logger
        self.vMaps = dict()
        
    @abstractmethod
    def addMap(self,map_id):
        """ Creates a new empty instance of a map"""
        pass

    @abstractmethod
    def removeMap(self,map_id):
        """ Deletes and instance of a map """
        pass
    
    def getMap(self,map_id):
        return self.vMaps[map_id]
        
    def loadMap(self, d_map):
        if(isinstance(d_map,str)):
            d_map = ast.literal_eval(d_map)
        map_id = d_map['MAP_ID']
        vMap = self.getMap(map_id)
        d_map.pop('MAP_ID',map_id)
        vMap.loadMap(d_map)
        vMap.updateMap()


class VirtualMapManagerLM(VirtualMapManager):
    """
      The VMapManager manages at the level of map.
      It has also the algorithms for error management.

    """


    def __init__(self,logger, server_msc):
        self.samples_list = dict()
        self.samples_list_center = dict()
        self.per_sample_eavg = dict()
        self.per_sample_estd = dict()
        self.global_error_avg = []
        self.global_error_std = []
        super(VirtualMapManagerLM, self).__init__(logger, server_msc)

    def addMap(self,map_id):
        self.vMaps[map_id] = virtualGridMapLM(self.logger)
        self.vMaps[map_id].map_id = map_id
        return self.vMaps[map_id]
        
    def removeMap(self,map_id):
        del self.vMaps[map_id]        
        return



    def addSetPoints(self, pointsOrigin, pointsDestiny, names, map_id, point_type, updateOrigin=False,
                     updateDestiny=False, updateModel=True, updateOccupancy=False):

        vMap = self.vMaps[map_id]

        pointsDestiny = np.array(pointsDestiny, dtype=np.float32)
        pointsOrigin = np.array(pointsOrigin, dtype=np.float32)

        if (not bool(pointsDestiny.shape)):
            s = (pointsOrigin.shape)
            pointsDestiny = np.empty(s)
            pointsDestiny.fill(np.inf)

        if (not bool(pointsOrigin.shape)):
            s = (pointsDestiny.shape)
            pointsOrigin = np.empty(s)
            pointsOrigin.fill(np.inf)

        for i in range(len(pointsOrigin)):
            coords_map = pointsOrigin[i]
            coords_lm = pointsDestiny[i]
            nid = names[i]
            vMap.addPoint(coords_map, coords_lm, point_type, nid, [1., 1.], updateModel=False)
            vMap.updateOrigin(nid, updateOrigin)
            vMap.updateDestiny(nid, updateDestiny)

        if(updateOccupancy):
            vMap.grid_map.update_grid(pointsDestiny, names)
        if (updateModel):
            vMap.updateMap()

    def generateGridMap(self,map_id,data,datamap,tags,orientation):
        vMap = self.vMaps[map_id]
        vMap.grid_map.generateGridMapCoordinates(data,tags,orientation)
        # There is a condition in which only 1 or 2 points are passed
        # If that is the case, we have to add at least 3 points to
        # generate a model. We pass 5.
        data=np.array(data)
        ds = data.shape
        if(ds[0]<3):
            tags,origincs,destinycs = vMap.grid_map.get4neighbors(tags[0])
            self.addSetPoints(origincs,destinycs, tags, map_id, PointType.calibrated, updateOccupancy=False)
        else:
            self.addSetPoints(datamap, data, tags, map_id, PointType.calibrated, updateOccupancy=False)

    def addSample(self,sample_name,centerPoint,datamap,datalm,tags,map_id, top_dist = 100):


        vMap = self.vMaps[map_id]
        # Project the new tags
        # Get canvas coords from grid_map

        bad_p = []
        for i,el in enumerate(tags):
            origincs = vMap.grid_map.map_coordinates_origin[el]
            point_dest = vMap.point_to_Destiny(origincs)
            m_dist = distance.euclidean(datalm[i],np.array((point_dest[0],point_dest[1],0.0),dtype=np.float32))
            # Calculate difference per point basis
            # if error>100 um per point OUT
            if(m_dist>top_dist):
                self.logger.info("Point "+el+" discarded:"+str(m_dist)+" um away")
                bad_p.append(i)
        # Remove bad points

        datalm = np.delete(datalm, bad_p,axis=0)
        datamap = np.delete(datamap, bad_p,axis=0)
        tags_aux = list(tags)
        for el in bad_p:
            tags_aux.remove(tags[el])
        tags = tags_aux

        if(datalm.shape[0]==0):
            self.logger.info("Sample wrongly assigned. NOT ADDING TO MAP.")
            return False
        # addSetPoints
        self.samples_list[sample_name] = tags
        self.samples_list_center[sample_name] = centerPoint
        self.addSetPoints(datamap, datalm, tags, map_id, PointType.calibrated, updateOccupancy=True)


        return True

    ###
    # crossvalidate and get error per point (add to vmap)
    # per sample error and total average and sd error
    def updateErrorByCV(self,map_id, num_neighs = 20):
        vMap = self.vMaps[map_id]
        per_sample_error_avg = []
        per_sample_error_std = []
        sample_name_list= list(self.samples_list_center.keys())
        problematic_points = True
        counter = 0
        self.global_error_avg = [np.inf]
        while(problematic_points):
            self.logger.info("ITERATION #"+str(counter))
            counter = counter+1
            for ind,centerpoint in enumerate(self.samples_list_center.values()):
                centerpoint = [centerpoint[1], centerpoint[0]]
                sample_name = sample_name_list[ind]
                # Get center point associated to sample
                # Get all closest 20 neighbors
                neighs,_ = vMap.CalibratedPtp._getNeighs(centerpoint, k =num_neighs, map_id = 2)
                # Get all tags we have associated to the sample
                m_tags = self.samples_list[sample_name]
                # remove them from the neighbors
                coords_map = vMap.CalibratedPtp.coordOrigin[neighs]
                bad_t = []
                for tag in m_tags:
                    tcoords = vMap.grid_map.map_coordinates_origin[tag]
                    for iind,ecm in enumerate(coords_map):
                        if(np.equal(tcoords[0],ecm[0]) and np.equal(tcoords[1],ecm[1])):
                            bad_t.append(iind)
                neighs = np.delete(neighs,bad_t)
                if(neighs.shape[0]<4):
                    continue
                # Calculate local transform using neighbors
                Hl,_ = vMap.CalibratedPtp.getLocalTransform(neighs)
                # Get predictions for all the points of my sample
                per_sample = []
                for tag in m_tags:
                    tcoords = vMap.grid_map.map_coordinates_origin[tag]
                    tmp = np.float32([tcoords[0], tcoords[1], 1.0])
                    predicted = np.dot(Hl, tmp.transpose())
                    predicted /= predicted[2]
                    predicted =  np.float32([predicted[0], predicted[1], 0.0])
                    # Get current actual coordinates
                    actual = vMap.getLandmark(tag,2)
                    actual = np.float32([actual[0], actual[1], 0.0])
                    # difference between each point and predictions
                    m_dist = distance.euclidean(predicted,actual)
                    #save error for each individual point in the map
                    vMap.list_errorDestiny[tag].append(m_dist)
                    per_sample.append(m_dist)

                avg_er = np.mean(np.ma.masked_array(per_sample))
                sd_er = np.std(np.ma.masked_array(per_sample))
                # calculate average error and std and save it for the sample
                bad_points = self.findBadPredictions(per_sample)
                if(np.any(bad_points)):
                    problematic_points = True
                    for bp in bad_points:
                         bp_id = m_tags[bp]
                         m_point = vMap.getMeasuresPoint(bp_id,2)
                         mp_list = []
                         for mp in m_point:
                             # calculate the error of predicted again
                             tcoords = vMap.grid_map.map_coordinates_origin[bp_id]
                             tmp = np.float32([tcoords[0], tcoords[1], 1.0])
                             predicted = np.dot(Hl, tmp.transpose())
                             predicted /= predicted[2]
                             predicted = np.float32([predicted[0], predicted[1], 0.0])
                             # Get current actual coordinates
                             actual = np.array([mp[0], mp[1], 0.0])
                             # difference between each point and predictions
                             m_dist = distance.euclidean(predicted, actual)
                             mp_list.append(m_dist)
                         # take the minimum
                         ind_min = np.argmin(mp_list)
                         # remove all the others
                         vMap.setMeasuresPoint(bp_id,2,m_point[ind_min])

                self.logger.info("Average error sample "+str(sample_name)+": "+ str(avg_er) + " +/-" + str(sd_er)+" um")
                per_sample_error_avg.append(avg_er)
                per_sample_error_std.append(sd_er)
                if sample_name in self.per_sample_eavg.keys():
                    self.per_sample_eavg[sample_name].append(avg_er)
                    self.per_sample_estd[sample_name].append(sd_er)
                else:
                    self.per_sample_estd[sample_name] = []
                    self.per_sample_eavg[sample_name] = []
                    self.per_sample_eavg[sample_name].append(avg_er)
                    self.per_sample_estd[sample_name].append(sd_er)

            # Final error is the mean of all per sample errors
            if(np.any(per_sample_error_avg)):
                self.global_error_avg.append(np.mean(per_sample_error_avg))
                self.global_error_std.append(np.std(per_sample_error_std))
                self.logger.info("Global error mean:"+str(self.global_error_avg[-1])+"+/-"+str(self.global_error_std[-1])+" um")
            problematic_points = self.global_error_avg[-1] != self.global_error_avg[-2]

        return

    def findBadPredictions(self,elist, tolerance = 10):
        bad_apples = True
        bp_list = []
        while(bad_apples):
            val_mean_bad = np.mean(elist)
            val_max = np.max(elist)
            ind_max = np.argmax(elist)
            good_list = np.delete(elist,ind_max)
            val_mean_good = np.mean(good_list)
            if (val_max-val_mean_good)>tolerance:
                # bad_point add_to_list
                bp_list.append(ind_max)
                elist = good_list
            else:
                bad_apples = False
        return bp_list


class VirtualMapManagerSEM(VirtualMapManager):
        """

        """
        scan_found = dict()
        scan_prepared = False

        def __init__(self,logger, server_msc):
            super(VirtualMapManagerSEM,self).__init__(logger,server_msc)

        ##########################################################################
        #       MAP Management
        ##########################################################################
        def cleanAll(self):
            self.vMaps = dict()

        def addMap(self,map_id):
            self.vMaps[map_id] = virtualGridMapSEM(self.logger)
            self.vMaps[map_id].map_id = map_id
            return self.vMaps[map_id]

        def addMapLM(self,map_id):
            self.vMaps[map_id] = virtualGridMapLM(self.logger)
            return self.vMaps[map_id]

        def removeMap(self,map_id):
            del self.vMaps[map_id]
            return

        def isempty(self,map_id):
            return self.vMaps[map_id].getTotalLandmarks()==0

        ##########################################################################
        #       Adding sets of points
        ##########################################################################
        def updateMapFromJSON(self, json_map, update = False):

            d1 = json_map["LM_SEM"]
            d2 = json_map["Canvas_SEM"]

            stmap = d2['MAP']
            map_df = pd.read_json(stmap)

            if np.any(map_df.TYPE == 'TARGET'):
                # Load new maps and replace them
                self.loadMap(d1)
                # Now we have to add all the landmarks from LM_SEM
                self.loadMap(d2)
                return
            elif np.any(map_df.TYPE == 'CALIBRATED') and update:
                v1 = self.getMap('LM_SEM')
                v2 = self.getMap('Canvas_SEM')
                # Now we have to see if there are only calibration coordinates or also targets
                v1.deleteCalibrations()
                v2.deleteCalibrations()
                map_df = map_df[map_df.TYPE == 'CALIBRATED']
                sem_coords = np.array(list(zip(map_df.COORDS_DESTINY_X, map_df.COORDS_DESTINY_Y, map_df.COORDS_DESTINY_Z)), dtype=np.float32)
                map_coords =  np.array(list(zip(map_df.COORDS_ORIGIN_X, map_df.COORDS_ORIGIN_Y, map_df.COORDS_ORIGIN_Z)), dtype=np.float32)
                self.addCalibratedPointsToMap(sem_coords, map_coords, list(map_df.LANDMARK))
            else:
                raise NonValidMap('Non valid map for updating')

        def addCalibratedPointsToMap(self, datasem, datamap, tags):
            self.addSetPointsFromMicroscope(tags, datasem, "LM_SEM", updateModel=False)
            self.addSetPoints(datamap, datasem, tags, "Canvas_SEM", PointType.calibrated, updateModel=False,
                                          updateOccupancy=True)
            v1 = self.getMap('LM_SEM')
            v2 = self.getMap('Canvas_SEM')

            v1.updateMap()
            v2.updateMap()
            return


        def applyTransform(self,M,map_id,map_to_update):
            self.vMaps[map_id].applyTransform(M,map_to_update)

        def addSetPoints(self,pointsOrigin,pointsDestiny,names,map_id,point_type,updateModel=True,updateOccupancy = False):
            """
                Given a sets of points in Origin, Destiny, their common landmark names and the map_id "Canvas_LM" or "LM_SEM"
                point_type 
            """
            vMap = self.vMaps[map_id]

            pointsDestiny = np.array(pointsDestiny,dtype=np.float32)
            pointsOrigin = np.array(pointsOrigin,dtype = np.float32)

            pD = pointsDestiny.shape
            pO = pointsOrigin.shape
            if(not bool(pD) or (pD[0] == 0)):
                pointsDestiny = np.empty(pO)
                pointsDestiny.fill(np.inf)

            if(not bool(pO) or (pO[0]==0) ):
                pointsOrigin = np.empty(pD)
                pointsOrigin.fill(np.inf)

            for i in range(len(pointsOrigin)):
                    coords_lm = pointsOrigin[i]
                    coords_sem = pointsDestiny[i]
                    nid = names[i]
                    vMap.addPoint(coords_lm, coords_sem, point_type, nid, [1., 1.], updateModel=False)
            if(updateModel):
                vMap.updateMap()
                
        ##########################################################################
        #        GRABBING FRAMES  and Mapper to front END
        ##########################################################################
        def prepare_scans(self, map_id,  percent = 1, recalibration = False):

            currentmap = self.getMap(map_id)
            self.scanning_map = currentmap
            letters_to_find = set()
            self.scan_prepared = False
            list_pointId = currentmap.getLandmarkIds()

            if len(list_pointId) == 0:
                return

            if len(list_pointId)<5:
                # Then select some random elements, starting from existent positions
                # Get existent letter

                pntid = list_pointId[0]
                all_labels =  currentmap.grid_map.getLabels()
                all_labels = [ label  for label in all_labels if '*' not in label ]  # Avoid corners
                all_labels = [ label  for label in all_labels if '+' not in label ]  # Avoid corners
                letters_to_find = self.getRandomLetters(all_labels, percent+0.1)
                distpid  = currentmap.grid_map.getCoordinatesGrid([pntid])
                to_find_map_coords = currentmap.grid_map.getCoordinatesGrid(letters_to_find)
                cal_tree = KDTree(to_find_map_coords)
                distances, indexes = cal_tree.query(distpid, k = 40, distance_upper_bound = np.inf)
                fletters_to_find = [letters_to_find[i] for i in indexes[0]]
                pOrigin = currentmap.grid_map.getCoordinatesGrid(fletters_to_find)
                self.addSetPoints(pOrigin,[],fletters_to_find,"Canvas_SEM",PointType.non_calibrated,False)
                self.scan_found = OrderedDict(zip(fletters_to_find, len(fletters_to_find) * [False]))
                return
            else:
                #####
                for pntid in list_pointId:
                    if(not currentmap.is_blocked(pntid)):
                        if(not currentmap.checkType(pntid,PointType.acquired) and not currentmap.checkType(pntid,PointType.target) ):
                            if(recalibration==False and not currentmap.checkType(pntid,PointType.calibrated)):
                                letters_to_find.add(pntid[0:2])
                            elif(recalibration == True):
                                letters_to_find.add(pntid[0:2])

                # grid map
                if percent < 1 or len(list(letters_to_find)) == 0:
                    letters_to_find = self.getRandomLetters(letters_to_find,percent)
                else:
                    letters_to_find = list(letters_to_find)
                # Sort by proximity to calibration points.
                # We took all the calibration points map coordinates, and all the letter map coordinates
                to_find_map_coords = currentmap.grid_map.getCoordinatesGrid(letters_to_find)

                calibrated_ids = currentmap.getLandmarkIDsByType(PointType.calibrated)
                calibrated_map_coords = currentmap.grid_map.getCoordinatesGrid(calibrated_ids)
                # Now, we find from all letters to find which one is the closest
                if np.any(calibrated_map_coords):
                    cal_tree = KDTree(calibrated_map_coords)
                else:
                    return
                dict_close = {}
                for ind, el in enumerate(to_find_map_coords):
                    dict_close[ind] = cal_tree.query(el)

                all_dist = dict_close.values()
                all_dist = sorted(all_dist, key=lambda x: x[0])
                closest_ind = all_dist[0][1]
                closest_val = calibrated_map_coords[closest_ind]
                find_tree = KDTree(to_find_map_coords)
                distances,indexes = find_tree.query(closest_val,k = len(letters_to_find), distance_upper_bound = np.inf )
                letters_to_find = [ letters_to_find[i] for i in indexes ]
                self.scan_found = OrderedDict(zip(letters_to_find, len(letters_to_find) * [False]))  # Create a map
            return
        def getListToScan(self):
            """
            Sort scan keys by proximity to calibration points
            
            :return: 
            """
            return self.scan_found.keys()

        def getRandomLetters(self,letters_to_find,percent):
            N = len(letters_to_find)
            total = list(range(0, N))
            np.random.shuffle(total)
            if(N>20):
                N = np.max([20.0,percent*N])
                N = int(N)
            total = total[0:N]
            total = np.sort(total)
            keys_selection = []
            for ind,el in enumerate(letters_to_find):
                if(ind in total):
                    keys_selection.append(el)
            return keys_selection

        def getCoordCenter(self,letter_id,map_id):
            currentmap = self.getMap(map_id)
            coords = currentmap.getLetterCoordinates(letter_id, 1)
            ecoords = np.zeros(coords.shape)
            ecoords[0] = coords[0]+currentmap.grid_map.spacing*0.5
            ecoords[1] = coords[1]+currentmap.grid_map.spacing*0.5
            ncoords = currentmap.point_to_Destiny(ecoords)
            return ncoords
            # letter_neighs = currentmap.getCornerNeighs(letter_id)
            # neighs = []
            # for sec_lett in letter_neighs:
            #     coords_stage = currentmap.getLetterCoordinates(sec_lett,2)
            #     neighs.append(coords_stage)
            # neighs = np.array(neighs)
            # cornerTop = neighs[0]
            # xmin = np.min(neighs[:,0])
            # xmax = np.max(neighs[:,0])
            # ymin = np.min(neighs[:,1])
            # ymax = np.max(neighs[:,1])
            # x_c = (xmax+xmin)*0.5
            # y_c = (ymax+ymin)*0.5
            # coord_center = (x_c,y_c,0.0)
            # return coord_center

        def getCoordsFile(self,dir_coord,re_filename):
                directories = glob.glob(dir_coord+'\*')
                xd1 = self.filterPick(directories,re_filename)
                if((not xd1)):
                    print("Error, coordinate files not detected")
                    c_file = directories[xd1[0]]
                    self.helper.readLM(c_file)
                    tags,indices = self.helper.unique_elements(self.helper.letters)
                    datalm = np.array(self.helper.coord_lm,np.float32)
                    datalm = datalm[indices]
                    datamap = np.array(self.helper.coord_map,np.float32)    ########
                    datamap = datamap[indices]
                    return (datalm,datamap,tags)
                else:
                    self.logger.info("Compute Line detection First.")
                    return

        def transformToNewData(self,itags,icoords,coords_id,map_id):
            vMap = self.vMaps[map_id]
            # Search coords from map

            ecoords  = vMap.getCoordsFromLandmarks(itags,coords_id) # 1 origin 2 destiny
            ecoords = np.squeeze(ecoords)
            if (len(ecoords)<3):
                return False
            ncor2 = (icoords[0][0:2], icoords[1][0:2], icoords[2][0:2])
            ncor = (ecoords[0][0:2], ecoords[1][0:2], ecoords[2][0:2])
            # calculate transform
            tform = tf.estimate_transform('affine', np.array(ncor), np.array(ncor2)) # 'affine'
            #tform = tf.AffineTransform(matrix=None, scale=None, rotation=tform.rotation, shear=None,  translation=tform.translation)
            self.applyTransform(tform.params, map_id, coords_id)
            return True


        def addSetPointsFromMicroscope(self,itags,icoords,map_id,updateModel = True):
            """
                Given points from a map, it transfers them as calibrated to the LM-SEM map
                or as acquired.
                Do not confuse this method with addSetPoints. 

                itags - letters or symbols associated to each coordinate
                icoords = coordinates from one of the maps
                map_id
                updateModel - since it is a costly operation to update all local regions of the map, we can decide to update the model
                later
            """
            vMap = self.vMaps[map_id]
            list_calibrated_names =[]
            list_calibrated = []
            list_acquired_names = []
            list_acquired= []

            # find if any of the tags matches with the tags we already have
            # That is, the point is calibrated, non-calibrated or target
            # We match it. Otherwise, we just add it as a landmark
            for ind,tag in enumerate(itags):
                if vMap.isin(tag) and not vMap.checkType(tag,PointType.acquired):
                    list_calibrated_names.append(tag)
                    list_calibrated.append(icoords[ind])
                else:
                    list_acquired_names.append(tag)
                    list_acquired.append(icoords[ind])

            if(list_calibrated_names):
                self.addSetPoints([],list_calibrated,list_calibrated_names,"LM_SEM",PointType.calibrated,updateOccupancy=True,updateModel=updateModel)
            if(list_acquired_names):
                self.addSetPoints([], list_acquired, list_acquired_names, "LM_SEM",PointType.acquired,updateModel=False,updateOccupancy=True)

            if(updateModel):
                vMap.updateMap()

        def blockPoint(self,point_id):
            # For each map managed, blocks position using the occupation map
            for el_map, m_map in self.vMaps.items():
                m_map.blockPoint(point_id)

        def unblockPoint(self, point_id):
            # For each map managed, unblocks position using the occupation map
            for el_map, m_map in self.vMaps.items():
                m_map.unblockPoint(point_id)

        def changeType(self, point_id, type):
            for el_map, m_map in self.vMaps.items():
                m_map.changeType(point_id, type)

        def updateErrorByCV(self, map_id, num_neighs = 20, tolerance = 50):
            vMap = self.vMaps[map_id]
            ## Get calibration points
            d_first, d_second, tags = vMap.getLandmarksByType("CALIBRATED")
            problematic_points = True
            error_list = []
            blocked_list = []
            blocked_error_list = []
            good_list = []
            self.logger.info("Finding bad apples in predictions!")
            for ind, tag in enumerate(tags):
                    if ind%5 == 0:
                        self.logger.info("# Iteration %s"% ind)
                    # Get all closest 20 neighbors for tag
                    neighs, ntags, distances = vMap.CalibratedPtp.getNeighs(d_second[ind,0:2], k = num_neighs, map_id=2)
                    if len(neighs)<4:
                        self.logger.warning("Not enough points to check quality of linear model.")
                        return
                    # Calculate local transform using neighbors (ONLY NEIGHBORS, YOU ARE NOT INCLUDED)
                    d1  = vMap.getCoordsFromLandmarks(ntags[1:],1)
                    d2 = neighs[1:]
                    H1, bi, ba = self.ransac_model(d1,d2)
                    # How bad is my linear model from lineal? correlations below xxx have to be removed
                    if H1 is None:
                        self.logger.warning("Cannot validate positions by cross_validation. It is not recommended to continue.")
                        return
                    # Get coordinates origin to predict
                    tcoords  = vMap.getCoordsFromLandmarks([tag],1)
                    tmp = np.float32([tcoords[0,0], tcoords[0,1], 1.0])
                    # Get predicted coordinates
                    predicted = np.dot(H1, tmp)
                    predicted /= predicted[2]
                    predicted = np.float32([predicted[0], predicted[1], 0.0])

                    # Get current current coordinates
                    actual = vMap.getLandmark(tag, 2)
                    actual = np.float32([actual[0], actual[1], 0.0])

                    # difference between each point and predictions
                    m_dist = distance.euclidean(predicted, actual)
                    # save error for each individual point in the map
                    if m_dist > tolerance : # should be in micrometers!
                        self.logger.info("#!!# BLOCKED: Position :"+tag+ " with values "+str(actual)+" has exceedeed minimum error. Error found to be: "+str(m_dist))
                        blocked_list.append(tag)
                        blocked_error_list.append(m_dist)
                    else:
                        # remove them from the neighbors
                        self.logger.info("Position :" + tag + " with values " + str(actual) + ". Error found to be: " + str(m_dist))
                        error_list.append(m_dist)
                        good_list.append(tag)

            avg_er =     np.mean(error_list)
            sd_er  =     np.std(error_list)
            self.logger.info("Average error sample " + str(avg_er) + " +/-" + str(sd_er) + " um")
            return blocked_list, blocked_error_list, good_list, error_list


        def ransac_model(self, data_origin, data_destiny, min_error = 5, tolerance = 20, max_iterations = 200, stop_at_goal=True):
            best_ac = tolerance
            best_ic = 0
            best_model = None
            seed = random.randrange(sys.maxsize)
            random.seed(seed)
            data_or = list(data_origin)
            n = len(data_or)
            goal_reached = False
            for sample_size in range(n, 4, -1):
                old_set = set()
                for i in range(max_iterations):
                    s = random.sample(range(n), int(sample_size))
                    ns = old_set - set(s)
                    if len(ns) == 0 and i>0 :
                        break
                    else :
                        old_set = set(s)
                    m = tf.estimate_transform('affine', data_origin[s,0:2], data_destiny[s,0:2]).params
                    ic = 0
                    ac_error = 0
                    for j in range(n):
                        data_origin[j,2] = 1
                        error = self.get_error(m, data_origin[j],data_destiny[j,0:2])
                        if error < tolerance:
                            ic += 1
                            if ic == 1 :
                                ac_error = error
                            else:
                                ac_error = (ac_error+error)*0.5
                    if ac_error < best_ac:
                        best_ic = ic
                        best_model = m
                        best_ac = ac_error
                        if best_ac < min_error:
                            goal_reached = True
                            break
                if goal_reached:
                    break
            return best_model, best_ic, best_ac

        def get_error(self,M, ipoint, opoint):
            predicted = np.dot(M, ipoint)
            predicted /= predicted[2]
            predicted = np.float32([predicted[0], predicted[1]])
            m_dist = distance.euclidean(predicted, opoint)
            return m_dist