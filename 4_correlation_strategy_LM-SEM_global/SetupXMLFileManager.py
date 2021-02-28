import numpy as np
import os.path
import json
import xml.etree.ElementTree as ET
import re
import numpy as np

class SetupXMLFileManager:
    """Read the Setup file"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()

    def find_and_replace_XML_tag(self,tag,value):
        self.root.findall(tag)[0].text = str(value)

    def replaceCoordinates(self,coords):
         # We assume the coordinates are in mm, and they need to be in umeters
         coords = np.array(coords,dtype='float32')*1e6
         # and the first 2 coordinates negative
         self.find_and_replace_XML_tag('SamplePreparation/ATLAS3DSamplePrepShapes/Stage_State/X',-coords[0])
         self.find_and_replace_XML_tag('SamplePreparation/ATLAS3DSamplePrepShapes/Stage_State/Y',-coords[1])
    #     self.find_and_replace_XML_tag('SamplePreparation/ATLAS3DSamplePrepShapes/Stage_State/Z',coords[2])

    def writeXML(self,path):
        try:
            self.tree.write(path)
        except IOError as err:
            self.print(err.message)

    def getXML(self):
        return ET.tostring(self.root)

#film = SetupXMLFileManager('C:\\Users\\JMS\\Documents\\msite\\MSite\\msite4A\\resources\\setups\\Atlas_3D_sample3.a3d-setup')
#coords = [ 0.06850099945,  0.09370700073, 0.0]
#film.replaceCoordinates(coords)
#film.writeXML('C:\\Users\\JMS\\Documents\\msite\\MSite\\msite4A\\resources\\setups\\Atlas_3D_sample3.a3d-setup2')