import xml.etree.ElementTree
import numpy as np

def read_mark_and_find_list(file):
    e = xml.etree.ElementTree.parse(file).getroot()
    pointlist  = []
    for maf in e.iter('MarkAndFindPoint'):
        tmp = maf.attrib
        pointlist.append([np.float(tmp["FieldXCoordinate"]), np.float(tmp["FieldYCoordinate"]), np.float(tmp["FieldZCoordinate"])])
    return pointlist

def write_mark_and_find_list(filename, points):
    with open(filename, "w") as f:
        f.write("""<?xml version="1.0" encoding="utf-8"?>\n""")
        f.write("""<MarkAndFindPointList xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n""")
        for pos in points:
            if points.shape[1]==3:
              f.write("""  <MarkAndFindPoint FieldXCoordinate="{0:.14f}" FieldYCoordinate="{1:.14f}" FieldZCoordinate="{2:.14f}" FieldRotation="0" />\n""".format(pos[0],pos[1],pos[2]))
            else:
              f.write("""  <MarkAndFindPoint FieldXCoordinate="{0:.14f}" FieldYCoordinate="{1:.14f}" FieldZCoordinate="{2:.14f}" FieldRotation="0" />\n""".format(pos[0],pos[1],0.0))

        f.write("""</MarkAndFindPointList>\n""")
        f.close()
# r"/Users/volkerhilsenstein/Downloads/{MarkAndFind}golgi_phenotypes.xml"