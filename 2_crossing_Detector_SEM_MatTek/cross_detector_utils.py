import sys, os
import time
import cv2
import numpy as np
import math
from scipy.signal  import convolve2d
from scipy.ndimage import label,sum
from skimage.transform import rotate as imrotate
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.segmentation import slic
from scipy.stats import mode
from collections import deque
import itertools
from common_analysis import canny, ridgeorient


def calculateOrientations(img):
        gradient,_ = canny(img,2)
        ORIENTIM, _ = ridgeorient(gradient, 1, 5, 5)
        segments = slic(img/255., n_segments=2500, sigma=1.5, compactness=0.08, start_label=0)
        num_labels = np.max(segments) + 1
        orientim_slic = np.copy(ORIENTIM)
        for i in range(num_labels):
            orientim_slic[np.where(segments == i)] = np.median(ORIENTIM[np.where(segments == i)])
        return orientim_slic
 
def projections(iswt, iorient, K = 20, inc = 1, aspace = None, arange = None):
        if (K < 4 or K > 1024):
            print('Invalid average value. Accepted values between 4 and half the size of your image. Setting default value.')
            K = 12

        if (inc > 90 or inc < 0):
            print('Invalid Delta, must be positive and less than 90')
            inc = 1
        # pad the image with zeros so we don't lose anything when we rotate.
        iLength, iWidth = iswt.shape
        iDiag = math.sqrt(iLength**2 + iWidth**2)
        LengthPad = math.ceil(iDiag - iLength) + 1
        WidthPad  =  math.ceil(iDiag - iWidth) + 1

        padIMG = np.zeros((iLength+LengthPad, iWidth+WidthPad))
        pad1 = int(math.ceil(LengthPad/2))
        pad2 = int(math.ceil(LengthPad/2)+iLength)

        pad3 = int(math.ceil(WidthPad/2))
        pad4 = int(math.ceil(WidthPad/2)+iWidth)
        padIMG[pad1:pad2, pad3:pad4] = iswt

        padIMGOR = np.zeros((iLength+LengthPad, iWidth+WidthPad))
        padIMGOR[pad1:pad2,pad3:pad4]= iorient
        #
        #  loop over the number of angles, rotate 90-theta (because we can easily sum
        #  if we look at stuff from the top), and then add up.  Don't perform any
        #  interpolation on the rotating.
        #
        #   -90 and 90 are the same, we must remove 90
        THETA = list(range(-90,90,inc))
        th = np.zeros(len(THETA))+np.inf
        if(arange):
            for ang in aspace:
                 k = ang+90
                 kplus = k+arange
                 kminus = k-arange
                 if(kplus>179):
                     kplus = 179
                 if(kminus<0):
                     kminus = 0
                 th[k:kplus] = THETA[k:kplus]
                 th[kminus:k] = THETA[kminus:k]
        else:
            th = THETA
        th = np.array(th,dtype =np.float32)*np.pi*(1/180.0)

        n = len(THETA)
        PR = np.zeros((padIMG.shape[1], n))
        M = padIMG # > 0

        iPL,iPW = padIMG.shape
        center = (iPL / 2, iPW / 2)
        for i in range(n):
            if(th[i]!=np.inf):
                final =oft(M,K, padIMGOR,th[i])
                Mt = cv2.getRotationMatrix2D(center, -THETA[i], 1.0)
                rotated = cv2.warpAffine(final, Mt, (iPL, iPW))
                PR[:,i] = (np.sum(rotated,axis=0))
            else:
                PR[:,i]=0

        PR[np.nonzero(PR<0)]=0.0
        PR = PR/iDiag
        PR = PR*10
        PR = np.multiply(PR,PR)
        PR = PR*0.1
        PR = PR/np.max(PR)
        return PR

def oft(M,K,L,ang, probabilistic = True):
        kernel = np.zeros((K,K))
        v_cos = math.cos(ang)
        v_sin = math.sin(ang)
        Mval = np.cos(2*(L-ang))
        if probabilistic:
            Mval = Mval*M
        count = 0
        for k in range(-int(K/2)-1,int(K/2)+2):
            ni = round(K/2+k*v_cos)
            nj = round(K/2+k*v_sin)
            if((ni>-1 and ni<K) and (nj>-1 and nj<K)):
                kernel[ni,nj]=1
                count +=1

        kernel = kernel/count

        cO = convolve2d(Mval, kernel, 'same')
        Or = np.zeros(M.shape)
        Or[np.nonzero(M)] = cO[np.nonzero(M)]
        return Or

def detectPeaksNMS(h, numpeaks = 1,threshold = None, nhood = None):
    """
    FindPEAKS Identify peaks in SOFT transform.
       PEAKS = detectPeaksNMS(H,NUMPEAKS) locates peaks in projection space. 
    NUMPEAKS specifies the maximum number of peaks to identify. PEAKS is 
    a Q-by-2 matrix, where Q can range from 0 to NUMPEAKS. Q holds
    the row and column coordinates of the peaks. If NUMPEAKS is 
    omitted, it defaults to 1.
    
    'Threshold' Nonnegative scalar.
               Values of H below 'Threshold' will not be considered
               to be peaks. Threshold can vary from 0 to Inf.
               Default: 0.5*max(H(:))

    'NHoodSize' Two-element vector of positive odd integers: [M N].
               'NHoodSize' specifies the size of the suppression
                neighborhood. This is the neighborhood around each 
                peak that is set to zero after the peak is identified.
               Default: smallest odd values greater than or equal to
                        size(H)/50.
    H is the output of the projections function. NUMPEAKS is a positive
    integer scalar.
    """
    # Set the defaults if necessary
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if nhood is None:
      nhood = np.array(h.shape)/(h.shape[0]*0.05)
      nhood = 2*np.array(nhood*0.5,dtype=np.int32) + 1 # Make sure the nhood size is odd.
      if nhood[0]==0 : nhood[0] = 1
      if nhood[1]==0 : nhood[1] = 1
        
    if not threshold:
      threshold = 0.5 *np.max(h.flatten())
    # initialize the loop variables
    done = False
    hnew = h.copy()
    nhood_center = np.array((nhood-1)/2,dtype=np.int32)+1
    peaks = []
    while not done:
        max_ind = np.argmax(hnew)  #ok
        p,q     = np.unravel_index(max_ind,hnew.shape)
        if hnew[p, q] >= threshold:
            if(q == 180 or q == 179):
                hnew[:,0:3] = np.flipud(hnew[:,0:3]) # Invert -89 and -88 to be like 90
            if(q == 0 or q == 1):
                hnew[:,178:180] = np.flipud(hnew[:,178:180]) # Invert -89 and -88 to be like 90
            
            peaks.append([p,q,hnew[p,q]]) # add the peak to the list
            p1 = p - nhood_center[0] 
            p2 = p + nhood_center[0]
            q1 = q - nhood_center[1]
            q2 = q + nhood_center[1]
                         
            # Create a square around the maxima to be supressed
            qq, pp  = np.meshgrid(range(q1+1,q2+1), range(max(p1+1,0),min(p2+1,h.shape[0])),indexing='ij')
            qq = qq.flatten()
            pp = pp.flatten()
 
            # For coordinates that are out of bounds in the theta
            # direction, we want to consider that is circular
            # for theta = +/- 90 degrees.
            theta_too_low = np.where(qq < 0)
            if theta_too_low : 
                qq[theta_too_low] = h.shape[1] + qq[theta_too_low]
    
            theta_too_high = np.where(qq > h.shape[1])
            if theta_too_high : 
                qq[theta_too_high] = qq[theta_too_high] - h.shape[1]
            # Convert to linear indices to zero out all the values.
            for ind,_ in enumerate(pp):
                    hnew[pp[ind],qq[ind]] = 0
            if(q == 180 or q == 179):
                hnew[:,0:3] = np.flipud(hnew[:,0:3]) # After supress, return the signal to normality
            if(q == 0 or q == 1):
                hnew[:,178:180] = np.flipud(hnew[:,178:180]) # Supress the complementary too
    
            done = (len(peaks) == numpeaks)
        else:
            done = True

    for ind,el in enumerate(peaks):
            peaks[ind][1]= el[1] - 90
 

    return peaks, hnew
	
def nonmaxsup1D(signal,npeaks,window, top):
    k = 0
    peaks = np.zeros((npeaks,2))
    num = 1.0
    while(npeaks>0):
        num = np.max(signal)
        indx    = np.argmax(signal)
        if(num < top): break
        peaks[k,0] = num
        peaks[k,1] = indx
        for i in range(indx-window,indx+window):
            if(i>1 and i<len(signal)): 
                signal[i]= 0
        npeaks = npeaks-1
        k=k+1
    return peaks	
	
def findCrossSequence(ipeaks,gridsize = None):
    # Checks which is the best combination of peaks that fits the gridsize
    # The allowed grid sequence can be one strike after another 
    # or just one line.
    # Peaks need to be tuples [position,angle,X]
    # Grid size must be in pixels. This findCrossSequence expects a separation of n pixels  ------------*----*--------
    # If we need n,m pixels (periodical)  ---*--*--------*--*------ needs modification
    # If gridsize is None, the pairing is omitted
	# Always return [] if error
    total_peaks = len(ipeaks)
    gpeaks = []
    if gridsize is None: return []
    
    # Get all possible combinations of peaks
    possible_pairs = list(itertools.combinations(ipeaks, 2))  
    error = []
    saved_group = []
    for pair in possible_pairs: 
        pair = np.array(pair)
        total_p = pair[pair.argsort(axis=0)[:,0]]
        terror= 0
        dist_p = np.abs(total_p[0,0]- total_p[1,0])
        good = 1
        dif_error = np.abs(dist_p - gridsize[0])/gridsize[0]
        terror= terror+dif_error;
        # first test, distance of spacing
        if(dif_error>0.35): good = 0
        
        if(good==1):
            if (np.abs(total_p[0,1]-total_p[1,1])>2): # second test, difference between angles not bigger than 3 deg
                good = 0
        
        if(good==1):
            saved_group.append(total_p)
            error.append(terror)
    
    if len(saved_group) == 0:
        return []
    # For each group check that they are not competing in the same distances
    k = 0
    good_group = []
    if(len(saved_group) == 1): return saved_group
   
    for i, g1 in enumerate(saved_group):
        for j in range(i+1,len(saved_group)):
            if(i!=j):
                g2 = saved_group[j]
                err_dif = np.abs(error[i]-error[j])
                if(err_dif>0.1): # We keep the set with minimum error
                    if(error[i]<error[j]): 
                        good_group.append(g1)
                    else:
                        good_group.append(g2)
                    
                else: #potential candidates, leave them
                    good_group.append(g1)
                    good_group.append(g2)

    if len(good_group)==0:
        return []
   
    # Now is time to select the BEST candidate
    # Is going to be the one that sums up the most
    sg = []
    for g in good_group:
        sg.append(np.sum(g[:,2]))
    maxind = np.argsort(np.array(sg))[::-1]
    return good_group[maxind[0]]

def enoughpeaks(pospeaks,negpeaks):
    """
    -1,-2,-3 if pos and neg are 0
         2 if neg good and pos bad
         1 if pos good and neg bad
         3 if both good
    """
    total_pos = np.squeeze(np.array(pospeaks)).shape[0]
    total_neg = np.squeeze(np.array(negpeaks)).shape[0]
    if(total_pos==0 and total_neg==0):
        return -1
    if (total_neg==0):
        if(total_pos<2): return -2
        else: return 1
    if(total_pos==0):
        if(total_neg<2): return -3
        else: return 2
    if(total_pos<2 and total_neg>=2): return 2
    if(total_pos>=2 and total_neg<2): return 1
    return 3

def predictGrid(best_peaks,R,gridsize=None):
    """
        Takes a group of peaks and tries
        to predict where the complementary angle peaks are
        based on the gridsize
    """
    error=0;
    goodpeaks = []
    if gridsize is None:
        return []
    best_peaks = np.squeeze(np.array(best_peaks))
    R = np.array(R)
    total_peaks = best_peaks.shape[0]
    if(total_peaks<2):
        return []

    angle = int(best_peaks[0,1]) # We assume that  the first is the highest signaling peak
    # normalize to same angle
    best_peaks[:,1]= angle 
    cangle = 0
    if(angle>0):
        cangle = angle - 90
    else : 
        cangle = angle+90
    c180angle = cangle+90
    
    signal = R[:,int(c180angle)].copy() # get complementary data  
    speaks = nonmaxsup1D(signal,6,int(gridsize[0]*0.5),0.01)

    if(speaks.shape[0]==0):
        return []
    fpeaks = np.zeros((6,3))
    for i in range(6):
        fpeaks[i,0] = speaks[i,1]
        fpeaks[i,1] = cangle
        fpeaks[i,2] = speaks[i,0]
    
    goodpeaks = findCrossSequence(fpeaks,gridsize)
    total_peaks = np.squeeze(np.array(goodpeaks)).shape[0]
    if total_peaks==0:
        # Could be that we have just one peak
        origin = fpeaks[0,0]
        max_l = R.shape[0]
        goodpeaks = fpeaks[:2]
        if fpeaks[1,0]<1e-6:
            return []
        direction = fpeaks[1,0] # Direction is second biggest peak
        if(direction>origin):
            goodpeaks[1,0] = origin+gridsize
        else:
            goodpeaks[1,0] = origin-gridsize
        
        if(goodpeaks[1,0]>max_l or goodpeaks[1,0]<0):
            goodpeaks = []
            return []
        
    return goodpeaks
	
def discardwrongpeaks(R,positivePairs,negativePairs,gridsize, verbose = True):
    """  Discard wrong peaks  
         Discard wrong peaks is based in the following facts:
          - In a grid you will find positive and negative angles
          - Make pairs of positive and negative angles. We group first positive, then negative
          - If we don't have enough for a square, we finish at that point.
      Otherwise,  we check that the pairs sum 90 degrees complementary, so we
      group by square. For each positive, it must exist a negative that
      complements, and vice versa.
       ------------------------------------------------------------------------
      Function ERRORS:
      - couldn't find enough peaks fitting grid conditions
    """
    goodpeaks = []
    error = 0    
    error = enoughpeaks(positivePairs,negativePairs)
    # Error can be 0, 1 or 2. I
    anglepos = np.array([ peak[1] for peak in positivePairs ])
    angleneg = np.array([ peak[1] for peak in negativePairs ])

    if(error<0) : return -1,[],[]   
    if error == 1:
        positivePairs = findCrossSequence(anglepos,gridsize)
        negativePairs = predictGrid(positivePairs,R,gridsize)
        if len(negativePairs)==0 or len(positivePairs)==0:
            return -1,[],[]
    if error == 2:
        negativePairs = findCrossSequence(angleneg,gridsize)
        positivePairs = predictGrid(negativePairs,R,gridsize)
        if len(negativePairs)==0 or len(positivePairs)==0:
            return -2,[],[]
    # We have 2 exceptions: 0 and -1, in that case, majority wins and is converted
    ### TODO: To be tested
    total_0 = np.sum(angleneg==0)
    total_1 = np.sum(anglepos==1)
    total_m1 = np.sum(angleneg==-1)
    total_90 = np.sum([anglepos==90,anglepos==89])  #We consider 90 and 89
    total_m89 = np.sum([angleneg==-90,angleneg==-89,angleneg==-88]) # We consider -89 and -88
    # We have to group angles in complementary sets:
    # -1,0,1 can show up together
    # -89,90 can also show up together
    # However, we only admit correspondences such as: (-1,0)+(90,89)  or  (1,2)+(-89,-88)
    # If we have many pairs, we only keep the majority
    if(total_0 > 0 and total_1 > 0) or (total_90 > 0 and total_m89 > 0) :
        if(total_90+total_0+total_m1 > total_m89+total_1): 
                # Remove all 1's and -89
                anglepos = anglepos[np.where(anglepos != 1)]
                angleneg = angleneg[np.where(angleneg != -89)]
                angleneg = angleneg[np.where(angleneg != -88)]
        else :  # Remove all 0's and 90's
                anglepos = anglepos[np.where(anglepos != 0)]
                angleneg = angleneg[np.where(angleneg != -1)]
                anglepos= anglepos[np.where(anglepos != 89)]
                anglepos = anglepos[np.where(anglepos != 90)]
    # To be tested
     
    # now make 90 degrees pairs
    # Take first positive angles and compare with negative if they add
    # 90+/-5
    pos_ang = np.unique(anglepos)
    neg_ang = np.unique(angleneg)
    good_angles_pos = set()
    good_angles_neg = set()

    for elp in pos_ang:
         for eln in neg_ang:
            nty = elp+np.abs(eln)
            if(nty>85 and nty<95): #good combination
                good_angles_pos.add(elp)
                good_angles_neg.add(eln)

    
    good_pos = [ positivePairs[ind] for ind,angle in enumerate(anglepos) if angle in good_angles_pos]
    good_neg = [ negativePairs[ind] for ind,angle in enumerate(angleneg) if angle in good_angles_neg]
        
    error = enoughpeaks(good_pos,good_neg)
    if error<0:
        return -3,[],[]
    
    positivePairs = findCrossSequence(good_pos,gridsize)
    negativePairs = findCrossSequence(good_neg,gridsize)
    
    if len(positivePairs)==0 and len(negativePairs)==0: 
        return -3,[],[]
    
    error = enoughpeaks(positivePairs,negativePairs)
   
    if error == 1:
        negativePairs = predictGrid(positivePairs,R,gridsize)
        if len(negativePairs)<1:
            return -1,[],[]
    if error == 2:
        positivePairs = predictGrid(negativePairs,R,gridsize)
        if len(positivePairs)<1:
            return -2,[],[]
    # get angles  
    positivePairs = np.squeeze(np.array(positivePairs))
    negativePairs = np.squeeze(np.array(negativePairs))    
    total_pos = positivePairs.shape[0]
    total_neg = negativePairs.shape[0]
    topval = 91
    topscore=0
    for elp in positivePairs:
        for eln in negativePairs:
            val = np.abs((elp[1] - eln[1])-90)
            score = elp[2]+eln[2]
            if( val < topval):
                fangpos = elp[1]
                fangneg = eln[1]
                topscore = score 
                topval = val
            elif val == topval:
                if score>topscore:
                    fangpos = elp[1]
                    fangneg = eln[1]
                    topscore = score 
                    topval = val
    
    nty = fangpos - fangneg
    

    for i in range(total_pos):
        positivePairs[i,1] = fangpos
    for i in range(total_neg):
        negativePairs[i,1] = fangneg
    
    if verbose: 
        print('Angle sum :'+str(nty))
    if(nty<(85) and nty>(60)) and verbose: print('The angle orientations are not 90 degrees. Adjust properly.')
    return 0,positivePairs,negativePairs
	
def getPeaks(img,img_p,gridsize, verbose = False):
    """
        Finding peaks by non maximum supression 
        select right peaks, then scan in upper and lower parts of the signal to
        get new weaker peaks
    """
    error = 0
    orientations = calculateOrientations(img)
    prj_pos = projections(img_p,orientations,aspace=[45], arange = 10)
    prj_neg = projections(img_p,orientations,aspace=[-45], arange = 10)


    peaks_pos, hnew = detectPeaksNMS(prj_pos, numpeaks = 6,threshold = 0.1, nhood = None)
    peaks_neg, hnew = detectPeaksNMS(prj_neg, numpeaks = 6,threshold = 0.1, nhood = None)  
    error,ppeaks,npeaks = discardwrongpeaks(prj_pos+prj_neg,peaks_pos,peaks_neg,gridsize, verbose) # based on the grid definition
    if error<0 :
        return np.array([]),np.array([]),[]
    return ppeaks,npeaks,(prj_neg+prj_pos)


def tlines(iimg_or, ipeaks_pos, ipeaks_neg):
    """ This MATLAB function takes an image and a set of peaks
    # It locates the peaks inside the function and it translate them
    # to lines in the image (slope and intercept)
    # (0,0) up left corner

    # Get image size, value of diagonal,value of padding, and angle relation
    # between Lenght and Width.
    """
    iLength, iWidth = iimg_or.shape
    iDiag = np.sqrt(iLength * iLength + iWidth * iWidth)
    LengthPad = int(iDiag - iLength) + 1
    WidthPad = int(iDiag - iWidth) + 1

    iimg = (iimg_or > 0).copy()  # Make it totally binary

    padIMG = np.full((iLength + LengthPad, iWidth + WidthPad), -1, dtype=np.float32)
    padIMG[int(LengthPad / 2):(int(LengthPad / 2) + iLength), int(WidthPad / 2):(int(WidthPad / 2) + iWidth)] = iimg
    top = padIMG.shape[1]

    # tIMG = np.full((iLength + LengthPad, iWidth + WidthPad), 1, dtype=np.float32)
    # tIMG[int(LengthPad / 2):(int(LengthPad / 2) + iLength), int(WidthPad / 2):(int(WidthPad / 2) + iWidth)] = iimg_or
    # tIMG = cv2.cvtColor(tIMG, cv2.COLOR_GRAY2RGB)

    lines = np.array([ipeaks_pos[:, 0], ipeaks_neg[:, 0]]).flatten()
    peaks = np.vstack([ipeaks_pos, ipeaks_neg])
    i = 0
    m = 0
    iPL, iPW = padIMG.shape
    center = (int(iPL / 2), int(iPW / 2))
    rlines = []
    for k in range(len(lines)):
        Mt = cv2.getRotationMatrix2D(center, -peaks[k, 1], 1.0)
        rotated = cv2.warpAffine(padIMG, Mt, (iPL, iPW),flags=cv2.INTER_LINEAR)  # bilinear, crop
        # rotatedColor = cv2.warpAffine(tIMG,Mt,(iPL,iPW),flags=cv2.INTER_LINEAR)
        j = 0
        posx1 = 0
        posx2 = top
        while (j < top):
            if rotated[j,int(lines[k])] > 0:
                posx1 = j
                break
            j = j + 1
        while (j < top):
            if rotated[j,int(lines[k])] < 0:
                posx2 = j
                break
            j = j + 1
        if (j > top):
            posx2 = top-1

        # Use for debug
        # cv2.line(rotatedColor, (int(lines[k]),int(posx1)), (int(lines[k]),int(posx2)), (255, 0, 0), 2)
        # plt.imshow(rotatedColor)
        # plt.pause(1)  # <-------
        # plt.waitforbuttonpress(0)
        # plt.close()
        Mt2 = cv2.getRotationMatrix2D(center, peaks[k, 1], 1.0)
        p1x = np.dot(Mt2,(int(lines[k]),int(posx1),1))
        p2x = np.dot(Mt2,(int(lines[k]),int(posx2),1))
        p1x[0] = np.ceil(p1x[0] - WidthPad / 2)
        p1x[1] = np.ceil(p1x[1] - LengthPad / 2)
        p2x[0] = np.ceil(p2x[0] - WidthPad / 2)
        p2x[1] = np.ceil(p2x[1] - LengthPad / 2)
        rlines.append([p1x, p2x])
    iimgf = cv2.cvtColor(iimg_or, cv2.COLOR_GRAY2BGR)
    for p in rlines:
        p1 = np.array(p[0], dtype=np.int32)
        p2 = np.array(p[1], dtype=np.int32)

        cv2.line(iimgf, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 2)

    return rlines, iimgf



def findIntersectionPoints(goodlines):
    goodlines = np.array(goodlines,dtype = np.float32)
    tl = goodlines.shape[0]
    # Find inner square intersection points for each line
    slopes = []
    intercepts = []
    for k in range(0,tl):
    # take all points related to that line
        line = goodlines[k]
        p1 = line[0]
        p2 = line[1]
        slope = (p2[1] - p1[1]) / ((p2[0] - p1[0])+1e-18)
        slope = np.round(np.rad2deg(np.arctan(slope)))
        if np.isnan(slope): slope = 0
        if slope == -90 or slope == 90:
            ind = np.argmin([np.abs(p1[0] - p2[0]), np.abs([p1[1] - p2[1]])])
            intercept = (p1[ind]+p2[ind])*0.5
        else:
            intercept= np.round((p1[1] - np.tan(np.deg2rad(slope)) * p1[0]))
        slopes.append(slope)
        intercepts.append(intercept)
    # Now I have all my lines analytically, find the intersections
    ipoints = []
    iorder = []
    for i in range(len(slopes)):
        for j in range(i,len(slopes),1):
            if  slopes[i]!=slopes[j]:
                if ((slopes[i] == 90 and slopes[j]==-90) or (slopes[j]==-90 and slopes[j]==90)):
                    continue
                elif slopes[i]==90 or slopes[i]==-90:
                    xp = intercepts[i]
                    yp =  (np.tan(np.deg2rad(slopes[j])) * xp) + intercepts[j]
                elif slopes[j] == 90 or slopes[j]==-90 :
                    xp = intercepts[j]
                    yp = (np.tan(np.deg2rad(slopes[i])) * xp) + intercepts[i]
                else:
                    xp = (intercepts[j] - intercepts[i]) / (np.tan(np.deg2rad(slopes[i])) - np.tan(np.deg2rad(slopes[j])))
                    yp = (np.tan(np.deg2rad(slopes[i])) * xp) + intercepts[i]
                ipoints.append([np.round(xp),np.round(yp)])
                iorder.append((i,j))
    return ipoints, iorder


def checkpoint(ic, point):
    [iLength, iWidth] = ic.shape
    checked = 1
    if (point[0] < 0): return 0
    if (point[1] < 0): return 0
    if (point[0] > iWidth-1): return 0
    if (point[1] > iLength-1): return 0
    return checked


def selectGridPoints(iimg, ipeaks_pos, ipeaks_neg):
    """
        This  function takes all the lines from an image
        and calculate the intersection between lines
        and the crossing point.
        A folder can be given optionally to save the image of the lines.
    """
    iLength, iWidth = iimg.shape
    angpos = ipeaks_pos[0, 1]
    angneg = ipeaks_neg[0, 1]
    cpoints = []
    fpoints = []
    iimgc = []
    imglines = []

    pixel_lines, imglines = tlines(iimg, ipeaks_pos, ipeaks_neg)

    fpoints, spoints =  findIntersectionPoints(pixel_lines)
    fpoints = np.array(fpoints, dtype = np.int32)
    fpoints = fpoints[fpoints[:,0].argsort()]
    cpoints, _ = findIntersectionPoints(np.array((fpoints[1:3],[fpoints[0],fpoints[3]]))) # middle point
    if fpoints is None:
        return [],[],[],[]
    iimgc = cv2.cvtColor(iimg, cv2.COLOR_GRAY2RGB)
    for i in range(len(fpoints)):
            if checkpoint(iimg,fpoints[i,:]):
                cv2.circle(iimgc,(fpoints[i,0],fpoints[i,1]),2,(255,0,0),-1)

    cpoints = np.array(cpoints[0],dtype=np.int32)
    if checkpoint(iimg, cpoints):
        cv2.circle(iimgc, (cpoints[0],cpoints[1]) , 2, (255, 255, 0), -1)

    return cpoints, fpoints, iimgc, imglines


def autoThreshold(nm, canthresh, balance = 0.8):
    m, n = nm.shape
    max_factor = 0.95 * np.max(nm)
    factor_a = max_factor
    value = 0
    msize = m * n
    max_pix = int(msize * canthresh)
    iter = 0
    while (value < max_pix and iter < 50):
        iter = iter + 1
        value = np.sum(nm>factor_a)
        factor_a = factor_a * 0.9
        if (factor_a < 1e-15):
            break

    c1 = 0
    alpha_1 = 0.01
    alpha_2 = 0.01
    inv = True
    iter = 0
    while (np.abs(value - max_pix) > 200 and iter < 20):
        value = np.sum(nm > factor_a)
        iter = iter + 1
        if (value < max_pix):
            if (inv):
                alpha_1 = 0.01
                inv = False
            factor_a = factor_a - alpha_1
            c1 = c1 + 1
            if (c1 == 2):
                alpha_1 = alpha_1 * 2
                c1 = 0
        else:
            if (not inv):
                alpha_2 = 0.01
                inv = True
            c1 = c1 - 1
            factor_a = factor_a + alpha_2
            if (c1 == -2):
                alpha_2 = alpha_2 * 2
                c1 = 0
    return nm>(factor_a*balance),factor_a