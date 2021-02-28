# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 21:07:53 2015

@author: JMS
"""

import json
import os
import os.path
import re
import subprocess
import sys
import threading
import glob
from os.path import basename,isfile,join
from os import listdir
import time
import numpy as np
import pandas as pd
#from image_an.lineDetector.gridLineDetector import GridLineDetector as GLD
#from image_an.readers import imageToStageCoordinates_LM
from occupancy_map import Map

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import xml.etree.ElementTree as ET
import logging

"""
 SET OF UTILITIES to search for directories, retrieve files and preferences.

"""
def getRootPath():
    start_dir = os.getcwd()
    for i in range(5):
        start_dir_old = start_dir
        start_dir, tail = os.path.split(start_dir_old)
        if tail == 'CLEMSite':
            return start_dir_old
    return []

def findFolder(init_folder, folderName):
    for root, dirs, files in os.walk(init_folder):
        for name in dirs:
            if name.endswith(folderName):
                return os.path.join(root, name)
    return init_folder

def getPreferencesFile():
    start_dir = getRootPath()
    folderPrefs = findFolder( start_dir, 'preferences')
    return os.path.join(folderPrefs,'user.pref')

def readPreferences(fname):
    try:
        pref = json.loads(open(fname).read())
    except json.decoder.JSONDecodeError as err:
        print('Bad format of preferences file.' + str(err))
        sys.exit()
    prefdict = pref['preferences'];
    return prefdict

def readXML(xmlfile):
    try:
        tree = ET.parse(xmlfile)
    except ET.ParseError as err:
        print(err)
    root = tree.getroot()
    data_coord = []
    data_tag = []
    data_coordmap = []
    for child in root.findall('GridPoint'):
        xcoord = child.get('FieldXCoordinate')
        ycoord = child.get('FieldYCoordinate')
        zcoord = child.get('FieldZCoordinate')
        data_coord.append([float(xcoord), float(ycoord), float(zcoord)])
    for child in root.findall('GridPointRef'):
        xcoord = child.get('FieldXCoordinateRef')
        ycoord = child.get('FieldYCoordinateRef')
        zcoord = child.get('FieldZCoordinateRef')
        data_tag.append(child.get('Map'))
        data_coordmap.append([float(xcoord), float(ycoord), float(zcoord)])
    return data_coord, data_tag, data_coordmap

def saveXML(xmlfile, tags, coordinatesRef, coordinatesMap):
    root = ET.Element("GridPointList")
    i = 0
    for e, e2 in zip(coordinatesMap, coordinatesRef):
        # Add a fictious line with our data from the mapping
        # print e, e2
        doc = ET.SubElement(root, "GridPointRef")
        doc.set("FieldXCoordinateRef", str(e[0]))
        doc.set("FieldYCoordinateRef", str(e[1]))
        if (len(e) > 2):
            doc.set("FieldZCoordinateRef", str(e[2]))
        else:
            doc.set("FieldZCoordinateRef", u"0.0")
        doc.set("Map", tags[i])
        doc = ET.SubElement(root, "GridPoint")
        doc.set("FieldXCoordinate", str(e2[0]))
        doc.set("FieldYCoordinate", str(e2[1]))
        if (len(e) > 2):
            doc.set("FieldZCoordinate", str(e[2]))
        else:
            doc.set("FieldZCoordinate", u"0.0")
        # doc = ET.SubElement(root, "MarkAndFindPicture")
        #  doc.set("filename", fname)
        #  doc.set("modality", mod)
        i = i + 1
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(xmlfile)
    return

def filterPick(myList, myString):
    """

    :rtype: string
    """
    pattern = re.compile(myString)
    indices = [i for i, x in enumerate(myList) if pattern.search(x)]
    return indices

def indent(elem, level=0, more_sibs=False):
    i = "\n"
    if level != 0:
        i = i + (level - 1) * "  "
    num_kids = len(elem)
    if num_kids != 0:
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
            if level:
                elem.text = elem.text + '  '
        count = 0
        for kid in elem:
            indent(kid, level + 1, count < num_kids - 1)
            count = count + 1
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        if more_sibs:
            elem.tail = elem.tail + '  '
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            if more_sibs:
                elem.tail = elem.tail + '  '



class MsiteHelper:

    letter_classifier = []

    coord_lm = []
    coord_map = []
    letters = []
    
    coord_sem = []
    letters_sem = []    
    logger = []

    def __init__(self, parent = None):    
        pass


    def setLetterClassifier(self, iclassifier):
        self.letter_classifier = iclassifier

    def read(self, fname):
        extension = os.path.splitext(fname)[1]
        if extension == '.xml':
            self.coord_ref, self.letters, self.coord_map = self.readXML(fname)
        elif extension == '.json':
            self.coord_ref, self.letters, self.coord_map = self.readJSON(fname)
        else:
            raise ValueError('Extension not recognised. Only xml and json')

    def readSEM(self,fname):
        extension = os.path.splitext(fname)[1]
        dummy = []
        if extension =='.txt':
            self.coord_sem, self.letters_sem, dummy = self.readTXT(fname)   
        elif extension == '.xml':
            self.coord_sem, self.letters_sem, dummy  = self.readXML(fname)
        elif extension == '.json':
            self.coord_sem, self.letters_sem, dummy  = self.readJSON(fname)
        else:
            raise ValueError('Extension not recognised. Only xml, txt and json')

    def readLM(self,fname):
        fname = str(fname)
        extension = os.path.splitext(fname)[1]
        if extension =='.txt':
            self.coord_lm, self.letters, self.coord_map = self.readTXT(fname)   
        elif extension == '.xml':
            self.coord_lm, self.letters, self.coord_map  = self.readXML(fname)
        elif extension == '.json':
            self.coord_lm, self.letters, self.coord_map  = self.readJSON(fname)
        else:
            raise ValueError('Extension not recognised. Only xml and json')

        
    def readJSON(self,jsonfile):
        data = json.loads(jsonfile)
        # then use as a dictionary
        return data
        
    def readXML(self,xmlfile):
        try:
            tree = ET.parse(xmlfile)
        except ET.ParseError as err:
            self.logger.error(err)
        root = tree.getroot()
        data_coord = []
        data_tag = []
        data_coordmap = []
        for child in root.findall('GridPoint'):
            xcoord = child.get('FieldXCoordinate')
            ycoord = child.get('FieldYCoordinate')
            zcoord = child.get('FieldZCoordinate')
            data_coord.append([float(xcoord), float(ycoord), float(zcoord)])
        for child in root.findall('GridPointRef'):
            xcoord = child.get('FieldXCoordinateRef')
            ycoord = child.get('FieldYCoordinateRef')
            zcoord = child.get('FieldZCoordinateRef')
            data_tag.append(child.get('Map'))
            data_coordmap.append([float(xcoord), float(ycoord), float(zcoord)])
        return data_coord, data_tag,data_coordmap
        
    def readTXT(self,txtfile, sep = ','):
        """
        Reads a file in the format
        Name, X, Y, Z
        Data_coord_map is returned as unknown (all -1),
        in order to fill map requirements data.
        :param txtfile:
        :return:
        """
        data_tags = []
        data_coord_lm = []
        data_coord_map = []
        with open(txtfile, 'r') as f:
            for line in f:
                point = np.array([],dtype = np.float32)
                refs = line.rstrip().split(sep)
                if(len(refs)==4):  # 3 coords, tag, name or 2 coords, tag, name. If no Z value
                    z_value = 1.0
                    tag = refs[2]
                    type = refs[3]
                else:
                    z_value = refs[2]
                    tag = refs[3]
                    type = refs[4]
                point = np.array([float(refs[0]),float(refs[1]),float(z_value)], dtype = np.float32)
                if type == 'GridPositionRef':
                    data_tags.append(tag)
                    data_coord_lm.append(point)
                elif type == 'GridPosition':
                    data_coord_map.append(point)
                else:
                    data_coord_map(np.array([-1,-1,-1]))  # We assume there is no map POSITIONING
        return np.array(data_coord_lm),data_tags,np.array(data_coord_map)

    def saveFile(self,fname,tags,coordref,coordmap):
        extension = os.path.splitext(fname)[1]
        if extension =='.txt':
            self.coord_lm = self.saveTXT(fname,tags,coordref,coordmap)   
        elif extension == '.xml':
            self.coord_lm = self.saveXML(fname,tags,coordref,coordmap)
        elif extension == '.json':
            self.coord_lm = self.saveJSON(fname)
        else:
            raise ValueError('Extension not recognised. Only xml,txt and json')

    def saveJSON(self,fname,data):
        with open(fname, 'w') as outfile:
            json.dump(data, outfile)
    
    def saveXML(self,xmlfile,tags,coordinatesRef,coordinatesMap):       
        root = ET.Element("GridPointList")
        i = 0
        for e,e2 in zip(coordinatesMap, coordinatesRef):
            # Add a fictious line with our data from the mapping
            doc = ET.SubElement(root, "GridPointRef")
            doc.set("FieldXCoordinateRef", str(e[0]))
            doc.set("FieldYCoordinateRef", str(e[1]))
            if(len(e)>2):
                doc.set("FieldZCoordinateRef", str(e[2]))
            else:
                doc.set("FieldZCoordinateRef", u"0.0")
            doc.set("Map", tags[i])            
            doc = ET.SubElement(root, "GridPoint")
            doc.set("FieldXCoordinate", str(e2[0]))
            doc.set("FieldYCoordinate", str(e2[1]))
            if (len(e) > 2):
                doc.set("FieldZCoordinate", str(e[2]))
            else:
                doc.set("FieldZCoordinate", u"0.0")
            #  doc = ET.SubElement(root, "MarkAndFindPicture")
            #  doc.set("filename", fname) 
            #  doc.set("modality", mod)
            i = i + 1
        self.indent(root)
        tree = ET.ElementTree(root)
        tree.write(xmlfile)
        return
        
    def indent(self,elem, level=0, more_sibs=False):
        i = "\n"
        if level!=0:
            i = i + (level-1)*"  "
        num_kids = len(elem)
        if num_kids!=0:
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
                if level:
                    elem.text = elem.text + '  '
            count = 0
            for kid in elem:
                self.indent(kid, level+1, count < num_kids - 1)
                count = count + 1
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            if more_sibs:
                elem.tail = elem.tail + '  '
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
                if more_sibs:
                    elem.tail = elem.tail + '  '
    
    def saveTXT(self,txtfile,tags,coordinatesRef,coordinatesMap):
        coordinatesMap = np.array(coordinatesMap)
        coordinatesRef = np.array(coordinatesRef)
        strn=""
        for i in range(len(tags)):
                if(coordinatesRef.shape[1]==2):
                    coord_z = 1.0
                else:
                    coord_z = coordinatesRef[i,2]
                strn += str(coordinatesRef[i,0])+","+str(coordinatesRef[i,1])+","+str(coord_z)+","+tags[i]+",GridPositionRef"
                if coordinatesMap.size > 0:
                    strn += str(coordinatesMap[i, 0]) + "," + str(coordinatesMap[i, 1]) + "," + str(coord_z) + "," + tags[i] + ",GridPosition"

        with open(txtfile, "w") as of:
            of.write(strn)




    ## This function must run in an independent thread
    @staticmethod
    def getFile(m_folder,re_tag):
        """
         Search for an image with the specified regular expression
         that was produced lately
        :param val:
        :return:
        """
        dname = m_folder
        onlyfiles = [ f for f in listdir(dname) if isfile(join(dname, f)) ]

        x = filterPick(onlyfiles,re_tag)
        if not x:
            return ""
        # List of files
        files_to_check = []
        for el in x:
            files_to_check.append(join(m_folder,onlyfiles[el]))
        # Get the filename of the image
        files_to_check.sort(key=lambda x: os.path.getmtime(x))
        return files_to_check[-1]

    ###################################### MATLAB runner######################################

    def lineDetector(self, id, input_im, outputdir, params_file, mode, infoname, ask = False):
        """
        Calls Matlab in background and starts a background thread to wait for it to finish

        :param input_im:
        :param outputdir:
        :param params_file:
        :param mode:
        :param matlab_dir:
        :param infoname:
        :param ask:
        :return:
        """
        image_name = basename(input_im)
        true_letter = 0
        # remove extension
        img_n = image_name[:-4]
        logfile = 'log_'

        input_im = os.path.normpath(input_im)
        outputdir = os.path.normpath(outputdir)

        self.logger.info("Calling line detector... ")
        self.logger.info("In "+outputdir)
        self.logger.info("------")
        GLD().runGLD(input_im, outputdir, params_file, infoname, mode)
        ### Launch thread for checking if process is finished


    def detectorReady(self, dir_sample, sample_data):
        ## if I can find the file others, then I have to read it, get the negative angle (2nd position)
        directories = glob.glob( dir_sample + '\*')
        xd = filterPick(directories, 'ld_')
        if (xd == []):
            raise Exception("Directory with Line Detection files not found.")

        search_dir = directories[xd[0]]
        xfiles = glob.glob(search_dir + '\*')
        x = filterPick(xfiles, 'cutpoints.csv')
        if (not x):
            raise Exception("Error, generated file cutpoints (.*cutpoints.csv) not found.")

        cutpoints_file = xfiles[x[0]]
        # Search swt
        x = filterPick(xfiles, '_swt')
        if (not x):
            raise Exception("Error, generated file swt_ not found.")

        swt_file = xfiles[x[0]]
        # Read negative and positive angle
        x = filterPick(xfiles, '_peaks.csv')
        if (not x):
            raise Exception("Error, generated file cutpoints (.*cutpoints.csv) not found.")

        peaks_file = xfiles[x[0]]

        x = filterPick(directories, 'info.txt')
        if (not x):
            raise Exception("Error, generated file cutpoints (.*cutpoints.csv) not found.")

        info_file = directories[x[0]]
        with open(info_file) as data_file:
            data = data_file.read()
        info = json.loads(data)
        df_peaks = pd.read_csv(peaks_file)

        # throw to a pandas dataframe and extract positive angle and negative angle
        ####
        angles = np.array(pd.unique(df_peaks['angle'].dropna()))
        sdata = {}
        sdata['angneg'] = np.min(angles)
        sdata['angpos'] = np.max(angles)
        sample_data['e_pattern'] = info['letter_center']
        sample_data['distsq'] = sample_data['distsq']/ float(info['PixelSize'])
        date = time.strftime("%d%m%Y-%H%M")

        ############ Prediction
        df_points = self.letter_classifier.predict_letters_from_picture(cutpoints_file, swt_file, "letters_" + date, sample_data)
        # The center point is marked as 2, the ones in the central square as 1, and if is not the central square as 0
        cpoint_df = df_points.where(df_points['center_point'] == 2).dropna()
        if np.any(cpoint_df):
            cpoint = int(cpoint_df.index.tolist()[0])
            sdata['letter']=df_points.loc[cpoint]['letter']
        else:
            cpoint = np.nan
            sdata['letter'] = df_points.loc[0]['letter']
        sdata['cpoint'] = cpoint
        sdata['tpoints']= len(df_points.where(df_points['in'] == 1).dropna())


        px = float(info['PositionX'])
        py = float(info['PositionY'])
        size = (int(info['Height']), int(info['Width']))
        #
        fpoints = np.array([[df_points.loc[i]['x'], df_points.loc[i]['y']] for i in range(len(df_points))])
        stage_coordinates = imageToStageCoordinates_LM(fpoints, (px, py), size, float(info['PixelSize']))
        df_points['cx_stage'] = stage_coordinates[:, 0]
        df_points['cy_stage'] = stage_coordinates[:, 1]
        ## add to df_points
        directories = glob.glob(dir_sample + '\*')
        xd = filterPick(directories, 'ld_')
        df_points.to_csv(directories[xd[0]] + '\\' + info['tag'] + '_final_points.csv')
        # Remove Non used points
        df_points = df_points[df_points.where(df_points['in'] == 1).notnull()]
        df_points = df_points.dropna().reset_index(drop=True)
        tags_cal = list(df_points['letter'])
        if not tags_cal:
            return False
        datamap_cal = Map.getMapCoordinates(tags_cal)
        datalm_cal = [[df_points.loc[i]['cx_stage'], df_points.loc[i]['cy_stage']] for i in df_points.index.tolist()]
        self.saveFile(dir_sample + '\\' + info['tag'] + "_fcoordinates_calibrate.xml",
                             tags_cal, datalm_cal, datamap_cal)
        folder_ld, _ = os.path.split(cutpoints_file)
        with open(folder_ld + '\\' + info['tag'] + "_impar.csv", 'w') as wr:
            wr.write(str(cpoint) + "\n")
            wr.write(str(sdata['angneg']) + "\n")
            wr.write(str(sdata['angpos']) + "\n")
            wr.write(str(sdata['tpoints']) + "\n")
            wr.write(str(sdata['letter']) + "\n")
        return sdata
    #####################################################################################END MATLAB LD
    def getError(self,log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        matches = re.findall("ERROR", "".join(lines))
        if(not matches):
            return 0
        else:
            # Search for error type II
            return 2 # Grab new image
            
    def unique_elements(self,seq, idfun=None):
        """
        unique(a, lambda x: x.lower())
        """
        # order preserving
        if idfun is None:
          def idfun(x): return x
        seen = {}
        result = []
        indices = []

        for item in seq:
            marker = idfun(item)
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
            indices.append(seq.index(item))
        return result,indices

    ################# LOGGING functionality #############
    def function_logger(self, log_file, file_level, console_level=None):

        logger = logging.getLogger(log_file)
        logger.setLevel(logging.DEBUG)  # By default, logs all messages

        if console_level != None:
            ch = logging.StreamHandler()  # StreamHandler logs to console
            ch.setLevel(console_level)
            ch_format = logging.Formatter('%(asctime)s - %(message)s')
            ch.setFormatter(ch_format)
            logger.addHandler(ch)

        fh = logging.FileHandler("{0}.log".format(log_file))
        fh.setLevel(file_level)
        fh_format = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)-8s - %(message)s')
        fh.setFormatter(fh_format)
        logger.addHandler(fh)
        self.logger = logger
        return logger

    def close_and_save_log(self):
        logging.shutdown()






