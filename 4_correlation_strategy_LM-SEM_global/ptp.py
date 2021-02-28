# -*- coding: utf-8 -*-

"""
Created on Wed Jan 21 16:15:21 2015

@author: JMS

Note : Use of kd tree?
"""
import numpy as np
import scipy.spatial as spt
import cv2
# import common.affineFit as af
from skimage import transform as tf




def transformPoint(pointA, origin, destiny, method= 'affine'):
    H,_ = getLocalTransform(origin,destiny,method)
    x = pointA[0]
    y = pointA[1]
    if H is not None:
        tmp = np.float32([x, y, 1.0])
        trh = np.dot(H, tmp.transpose())
        trh /= trh[2]
        return np.array([trh[0], trh[1], 0.0])
    else:
        return (np.array([-np.inf, -np.inf, -np.inf]))


def getLocalTransform(newOrigin,newDestiny,method='affine'):
    # FIRST NORMAL
    if (method == 'affine'):
        try:
            # if(self.matrixrank(self.coordRef[neighs])>0):
            #hn = af.Affine_Fit(newOrigin, newDestiny)
            if newOrigin.shape[0]< 5:
                transform = tf.estimate_transform('similarity', newOrigin, newDestiny)
            else:
                transform = tf.estimate_transform('affine', newOrigin, newDestiny)
            hn = transform.params
            # hn,mask = cv2.findHomography(newOrigin,newDestiny)
            message = "Using affine fit with " + str(newOrigin.shape) + " neighbors."
            return hn, message
        except np.linalg.linalg.LinAlgError as err:
            return [], err.message
        except FloatingPointError as err:
            return [], err.message

    else:
        if newOrigin.shape[0] < 10:
            transform = tf.estimate_transform('similarity', newOrigin, newDestiny)
            hn = transform.params
        else:
            hn = cv2.findHomography(newOrigin, newDestiny)  # global are maintained
    return hn


################################################################################
class LocalArea:

    def __init__(self):
       self.clearArea()


    @staticmethod
    def dist(x, y):
        if (x[0] == np.inf or x[1] == np.inf or y[0] == np.inf or y[1] == np.inf):
            return np.inf
        else:
            return np.sqrt(np.sum((x - y) ** 2))

    ### Only possible when there is enough data
    def updateRadius(self,origin,destiny):
        #If we get some points...
        if (self.radius_origin == 0.0):
            self.radius_origin = 2 * self.dist(origin[0], origin[1])
            self.radius_destiny = 2 * self.dist(destiny[0], destiny[1])
            self.conversion_ratio = self.radius_origin / self.radius_destiny

    def getErrorRetro(self):
        return self.list_errorRetro

    def getTransform(self,inverse=False):
        if(inverse):
            return self.H_inv
        else:
            return self.H

    def clearArea(self):
        self.radius_origin = 0.0
        self.radius_destiny = 0.0
        self.H = None
        self.H_inv = None
        self.neighs = None
        self.conversion_ratio = 1
        self.list_errorRetro = np.zeros(2)


class PointToPoint:
        """ Class that creates local homographies
            The local homographies are based in two criteria:
                - Takes all points from a given radius
                - There must be a minimum of 5 points
                - If the condition is not fulfilled, a global homography is used
                
            Thus, this must be initialized with:
                	- Ref are coordinates from space 1 (6 at least)
                  - Map are coordinates from space 2 (6 at least)
                  - expected radius
        """
        coordOrigin = None
        coordDestiny = None
        tags = None
        Hg = None
        Hg_inv = None
        treeOrigin = None
        treeDestiny = None
        local = False
        def __init__(self):
            pass


        def reset(self):
            self.coordOrigin = None
            self.coordDestiny = None
            self.tags = None
            self.Hg = None
            self.Hg_inv = None
            self.treeOrigin = None
            self.treeDestiny = None
            self.local = False


        def updateGlobal(self, origin, destiny, tags):
            if (len(origin)<50):
                transform = 'affine'
                if len(origin<6):
                    transform = 'similarity'
                try:
                    transf = tf.estimate_transform(transform, origin, destiny)
                    self.Hg = transf.params
                    self.Hg_inv = transf._inv_matrix
                    self.local = False
                except FloatingPointError as err:
                    print ("Warning :" + str(err))
                    print ("You need to add more points!")
                    self.warning_transformation = "COPLANAR"
            else:
                self.Hg, mask = cv2.findHomography(origin, destiny)  # global are maintained
                self.Hg_inv, mask = cv2.findHomography(destiny, origin)
                self.local = True
            self.coordDestiny = destiny
            self.coordOrigin = origin
            self.tags = tags
            self.treeOrigin = spt.KDTree(origin)
            self.treeDestiny = spt.KDTree(destiny)


        def updateLocalArea(self,local_area,ref_origin):
            """

            """
            if(not local_area):
                return
            if(local_area.radius_origin == 0):
                local_area.updateRadius(self.coordOrigin[0:2],self.coordDestiny[0:2])

            local_area.neighs,d = self._getNeighs(ref_origin[0:2], local_area.radius_origin)
            local_area.H,message = self.getLocalTransform(local_area.neighs)
            local_area.H_inv,message = self.getLocalTransform(local_area.neighs, inverse = True)



        def getNeighs(self, pRef, usrad=0, k = 20, map_id = 1):

             neighs,distances = self._getNeighs(pRef,usrad,k+1,map_id) # k plus you!
             m_tags =list(self.tags[i] for i in neighs)
             if(map_id==1):
                return self.coordOrigin[neighs], m_tags,distances
             else:
                return self.coordDestiny[neighs], m_tags,distances


        def _getNeighs(self, pRef, usrad=0, k = 20, map_id = 1):
            if(map_id == 1):
                m_tree = self.treeOrigin
            else:
                m_tree = self.treeDestiny

            if (usrad == 0):
                # Get closest 25
                distances,neighs = m_tree.query(pRef,k)
                if(len(neighs)<3):
                        return [pRef],[] # Return own point
                else:
                    ind = 0
                    for ind,el in enumerate(distances):
                        if np.isinf(el):
                            return neighs[0:ind], distances[0:ind]
                    return neighs[0:ind+1],distances[0:ind+1]
            # Take a ball with a determined radius

            neighs = m_tree.query_ball_point(pRef, usrad)
            # We need at least 5 points for an homography, otherwise we just take the global
            if (len(neighs) > 4):
                try:
                    # your code that will (maybe) throw
                    # if(self.matrixrank(self.coordRef[neighs])>0):
                    # Now we calculate the new homography
                    return neighs,[]
                    # else:
                    #    return range(len(self.coordRef))
                except np.linalg.linalg.LinAlgError as err:
                    print(err.message)
            else:
                return [pRef],[]

        def getLocalTransform(self,neighs,inverse=False):
          """
              Given a point, we take all the neighbor coordinates in a radius.
              If there are less than 6 points, the global homography is used.
          """
          neighs = np.array(neighs,dtype=np.uint)
          if( neighs.shape[0] < 4 ):
              message = "Using GLOBAL homography"
              if(inverse==True):
                  return self.Hg_inv,message
              else:
                return self.Hg,message

          newOrigin = self.coordOrigin[neighs]
          newDestiny = self.coordDestiny[neighs]
          # FIRST NORMAL
          if(inverse ==False):
                hn, message = getLocalTransform(newOrigin,newDestiny, 'affine')
                if len(hn) == 0:
                        return [], message
                else:
                        message =  "Using Local homography with "+str(len(neighs))+" neighbors."
                        return hn, message

          else:
                hn, message = getLocalTransform(newDestiny, newOrigin, 'affine')
                if len(hn) == 0:
                      return [], message
                else:
                      message = "Using Local homography with " + str(len(neighs)) + " neighbors."
                      return hn, message


def matrixrank(A,tol=1e-8):
    """
    colinearity
            
    """
    s = np.linalg.svd(A,compute_uv=0)
    return sum( np.where( s>tol, 1, 0 ))

def unitvector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def anglebetween(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unitvector(v1)
    v2_u = unitvector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
