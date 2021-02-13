import cv2
import numpy as np
import math
from scipy.signal  import convolve2d
import matplotlib.pyplot as plt
from collections import deque
from skimage.segmentation import slic
from skimage import morphology
import random
from scipy.ndimage import label,sum
from functools import reduce


# Many functions have been adapted from Peter Kovesi : https://www.peterkovesi.com/matlabfns/


def plotPoints(img,points, color = 'red', size=10):
    implot = plt.imshow(img)
    # put a blue dot at (10, 20)
    points_x = points[:,0]
    points_y = points[:,1]
    plt.scatter([points_x], [points_y],c=color,s=size)
    plt.show()

def plotHist(img):
    # hist,bins = np.histogram(img.flatten(),256,[0,256])

    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

def normalise(im, reqmean = 0, reqvar = 1):
  im = np.array(im,dtype = np.float32)
  #im = im - np.mean(im)
  #im = im / np.std(im)
  # n = reqmean + im * np.sqrt(reqvar);
  return im

def canny(i_image,isigma):
    image = gaussfilt(i_image,isigma)
    Ix,Iy = derivative5(image)
    Ix_2 = np.multiply(Ix,Ix)
    Iy_2 = np.multiply(Iy,Iy)
    gradient = np.sqrt(Ix_2 + Iy_2)   # Gradient magnitude.
    orientation = np.arctan2(-Iy, Ix)                # Angles -pi to + pi.
    orientation[orientation<0] = orientation[orientation<0]+np.pi;             # Map angles to 0-pi.
    orientation = orientation*180/np.pi;
    return gradient,orientation

def gaussfilt(img,sigma):
    sze = int(math.ceil(6*sigma))
    if(sze%2 == 0):
            sze = sze+1
    h = fspecial_gauss2D((sze,sze),sigma)
    # conv2(image, mask) is the same as filter2(rot90(mask,2), image)
    image = convolve2d(img,h,'same')
    return image

def fspecial_gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """

        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

def derivative5(i_image):
    # 5 tap 1st derivative cofficients.  These are optimal if you are just
    # seeking the 1st derivatives
    # Copyright (c) 2010 Peter Kovesi
    p = np.array([0.037659,0.249153,0.426375,0.249153,0.037659], dtype = np.float32)
    d1 =np.array([0.109604,0.276691,0.000000,-0.276691,-0.109604],dtype = np.float32)

    a =  p[:,np.newaxis]*d1.transpose()
    b =  d1[:,np.newaxis]*p.transpose()
    Ix = convolve2d(i_image,a,'same')
    Iy = convolve2d(i_image,b,'same')
    return Ix,Iy

def floodfill(bw, r, c, N):
    filled = np.zeros(bw.shape)
    theStack = deque(zip(r, c))
    m, n = bw.shape
    while len(theStack) > 0:
        x, y = theStack.pop()
        if x < 0:
            x = 0
        if x >= n:
            x = n - 1
        if y < 0:
            y = 0
        if y >= m:
            y = m - 1
        if filled[x, y] == 1:
            continue
        if bw[x, y] == 0:
            continue
        filled[x, y] = 1

        theStack.append((x + 1, y))  # right
        theStack.append((x - 1, y))  # left
        theStack.append((x, y + 1))  # down
        theStack.append((x, y - 1))  # up
        if (N == 8):
            theStack.append((x + 1, y + 1))  # d right
            theStack.append((x - 1, y - 1))  # d left
            theStack.append((x - 1, y + 1))  # down
            theStack.append((x + 1, y - 1))  # up

    return filled


class Pixel:
    value = 0
    i = 0
    j = 0
    distance = 0
    label = 0

    def __init__(self,distance,i,j,label):
        self.distance = distance
        self.i = i
        self.j = j
        self.label = label

def propagate(img,mask,seeds,ilambda):
    labels_out = np.copy(seeds)
    dists = np.full(img.shape,np.inf)
    dists[seeds>0] = 0
    pq = deque([])
    total_seeds = seeds.max()+1
    for i in range(1,total_seeds):
        # Get all pixel coordinates from pixels that are seeds
        listpx, listpy = np.where(seeds==i)
        for x,y in zip(listpx,listpy):
                push_neighs_on_queue(pq,0.0,x,y ,img,ilambda,i,labels_out, mask)


    while(len(pq)>0):
        p = pq.popleft()
        if(dists[p.i,p.j]>p.distance):
            dists[p.i,p.j] = p.distance
            labels_out[p.i,p.j] = p.label
            push_neighs_on_queue(pq, p.distance,p.i,p.j, img, ilambda, labels_out[p.i,p.j], labels_out, mask)

    return dists,labels_out

def clamped_fetch(img,i,j):
    m,n = img.shape
    if i < 0:
        i = 0
    if i >= n:
        i = n-1
    if j < 0:
        j = 0
    if j >= m:
        j = m-1
    return img[i,j]

def difference(img,i1,j1,i2,j2,ilambda):
     pixel_diff = 0
     #s1 = integrate(ii,i1-1,j1-1,i1+1,j1+1)
     #s2 = integrate(ii,i2-1,j2-1,i2+1,j2+1)
     #pixel_diff = np.abs(s1-s2)
     dEucl = (i1-i2)*(i1-i2) + (j1-j2)*(j1-j2)
     #fdist =np.sqrt((pixel_diff * pixel_diff +dEucl*dEucl*ilambda*ilambda)) # / (1.0 +ilambda ))
     return  int(dEucl*ilambda)
    #return np.sqrt((pixel_diff * pixel_diff +ilambda *dEucl) / (1.0 +ilambda ))
   #return (sqrt(pixel_diff * pixel_diff + (fabs((double) i1 - i2) + fabs((double) j1 - j2)) * lambda * lambda ));

def push_neighs_on_queue(pq,distance,i,j,img,ilambda,label, labels_out, mask):
  #  4-connected
  m,n = img.shape
  if (i > 0):
    val  = labels_out[i-1,j]
    if (val==0 and mask[i-1, j]>0):
        delta_d = difference(img, i, j, i-1, j, ilambda)         # if the neighbor was not labeled, do pushing
        pix = Pixel(distance + delta_d, i-1, j, label)
        pq.append(pix)
  if (j > 0):
    val = labels_out[i,j-1]
    if  val==0 and mask[i, j-1]!=0 :
        delta_d = difference(img,i,j,i,j-1,ilambda)
        pix = Pixel(distance + delta_d, i, j-1, label)
        pq.append(pix)
  if i<(n-1):
    val =  labels_out[i+1,j]
    if (val==0 and mask[i+1, j]!=0) :
        delta_d = difference(img, i, j, i+1, j , ilambda)
        pix = Pixel(distance + delta_d, i+1, j , label)
        pq.append(pix)
  if (j < (m-1)):
    val = labels_out[i,j+1]
    if val==0 and (mask[i, j+1]!=0):
        delta_d = difference(img, i, j, i, j + 1, ilambda)
        pix = Pixel(distance + delta_d, i, j + 1, label)
        pq.append(pix)
  # 8-connected
  if (i > 0) and (j > 0):
    val = labels_out[i-1,j-1]
    if(val==0 and mask[i-1, j-1]!=0):
        delta_d = difference(img, i, j, i-1, j - 1, ilambda)
        pix = Pixel(distance + delta_d, i-1, j - 1, label)
        pq.append(pix)
    if (i < (n-1) and  (j > 0)):
        val=labels_out[i+1,j-1]
        if (val==0 and (mask[i+1, j-1])!=0):
            delta_d = difference(img, i, j, i+1, j - 1, ilambda)
            pix = Pixel(distance + delta_d, i+1, j - 1, label)
            pq.append(pix)
    if (i > 0) and j < (m-1):
        val =labels_out[i-1,j+1]
        if (val==0 and mask[i-1, j+1]!=0 ):
            delta_d = difference(img, i, j, i-1, j + 1, ilambda)
            pix = Pixel(distance + delta_d, i-1, j + 1, label)
            pq.append(pix)
    if (i < (n-1) and j < (m-1)):
        val=labels_out[i+1,j+1]
        if val==0 and (mask[i+1, j+1]!=0):
            delta_d = difference(img, i, j, i+1, j + 1, ilambda)
            pix = Pixel(distance + delta_d, i+1, j + 1, label)
            pq.append(pix)
    return

def integral_image(x):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    x : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image / summed area table.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    return x.cumsum(1).cumsum(0)

def integrate(ii, r0, c0, r1, c1):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Integral (sum) over the given window.

    """
    S = 0

    S += clamped_fetch(ii,r1,c1)

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += clamped_fetch(ii,r0-1,c0-1)

    if (r0 - 1 >= 0):
        S -= clamped_fetch(ii,r0-1,c1)

    if (c0 - 1 >= 0):
        S -= clamped_fetch(ii,r1,c0-1)

    return S

def softmax(y):
    s = np.exp(y)
    y_prob = s / np.sum(s)
    return y_prob

def remove_borders(img,border):
    # remove borders
    m,n = img.shape
    img[:border, :] = 0
    img[-border:, :] = 0
    img[:, :border] = 0
    img[:, -border:] = 0
    return img

def ridgeorient(im,gradientsigma,blocksigma,orientsmoothsigma, rel = 0.01):
        # Arguments:  im                - A normalised input image.
        #             gradientsigma     - Sigma of the derivative of Gaussian
        #                                 used to compute image gradients.
        #             blocksigma        - Sigma of the Gaussian weighting used to
        #                                 sum the gradient moments.
        #             orientsmoothsigma - Sigma of the Gaussian used to smooth
        #                                 the final orientation vector field.
        #                                 Optional: if ommitted it defaults to 0

        # Returns:    orientim          - The orientation image in radians.
        #                                 Orientation values are +ve clockwise
        #                                 and give the direction *along* the
        #                                 ridges.
        #             reliability       - Measure of the reliability of the
        #                                 orientation measure.  This is a value
        #                                 between 0 and 1. I think a value above
        #                                 about 0.5 can be considered 'reliable'.
        #                                 reliability = 1 - Imin./(Imax+.001);
        #             coherence         - A measure of the degree to which the local
        #                                 area is oriented.
        #                                 coherence = ((Imax-Imin)./(Imax+Imin)).^2;
        rows,cols = im.shape

        # Calculate image gradients.
        sze = int(np.fix(6*gradientsigma))
        if(sze%2 == 0):
            sze = sze+1
        h = fspecial_gauss2D((sze,sze),gradientsigma)
        fx,fy = np.gradient(h)  # Gradient of Gausian.

        Gx = convolve2d(im, fx,'same') # Gradient of the image in x
        Gy = convolve2d(im, fy, 'same') # ... and y

       # Estimate the local ridge orientation at each point by finding the
       # principal axis of variation in the image gradients.

        Gxx = np.multiply(Gx,Gx)       # Covariance data for the image gradients
        Gxy = np.multiply(Gx,Gy)
        Gyy = np.multiply(Gy,Gy)

        # Now smooth the covariance data to perform a weighted summation of the  data.
        sze = int(np.fix(6*blocksigma))
        if(sze%2 == 0):
            sze = sze+1
        h = fspecial_gauss2D((sze,sze),blocksigma)
        Gxx = convolve2d(Gxx, h,'same');
        Gxy = 2*convolve2d(Gxy,h,'same');
        Gyy = convolve2d(Gyy,h,'same');

        # Analytic solution of principal direction
        Gxy_2 = np.multiply(Gxy,Gxy)
        Gm = Gxx-Gyy
        Gm = np.multiply(Gm,Gm)
        denom = np.sqrt(Gxy_2 + Gm) + np.spacing(1)
        sin2theta = np.divide(Gxy,denom)            # Sine and cosine of doubled angles
        cos2theta = np.divide(Gxx-Gyy,denom)

        sze = int(np.fix(6*orientsmoothsigma))
        if(sze%2 == 0):
            sze = sze+1
        h = fspecial_gauss2D((sze,sze),orientsmoothsigma)

        cos2theta = convolve2d(cos2theta,h,'same')# Smoothed sine and cosine of
        sin2theta = convolve2d(sin2theta,h,'same'); # doubled angles

        orientim = np.pi/2 + np.arctan2(sin2theta,cos2theta)/2;

        # Calculate 'reliability' of orientation data.  Here we calculate the
        # area moment of inertia about the orientation axis found (this will
        # be the minimum inertia) and an axis  perpendicular (which will be
        # the maximum inertia).  The reliability measure is given by
        # 1.0-min_inertia/max_inertia.  The reasoning being that if the ratio
        # of the minimum to maximum inertia is close to one we have little
        # orientation information.

        Imin = (Gyy+Gxx)/2
        Imin = Imin - np.multiply((Gxx-Gyy),cos2theta)/2 - np.multiply(Gxy,sin2theta)/2
        Imax = Gyy+Gxx - Imin

        reliability = 1 - np.divide(Imin,(Imax+.001))
        # aux = Imax+Imin
        # aux = np.multiply(aux,aux)
        # coherence = np.divide((Imax-Imin),aux)

        # Finally mask reliability to exclude regions where the denominator
        # in the orientation calculation above was small.  Here I have set
        # the value to 0.001, adjust this if you feel the need
        reliability = np.multiply(reliability,(denom>rel))
        return orientim,reliability

def SWT(i_img, edgeImage, orientim, stroke_width=20, angle=np.pi / 6):

    orientim = np.radians(orientim)
    im = gaussfilt(i_img, 1)
    Ix, Iy = derivative5(im)
    Ix_2 = np.multiply(Ix, Ix)
    Iy_2 = np.multiply(Iy, Iy)
    g_mag = np.sqrt(Ix_2 + Iy_2)  # Gradient magnitude.
    Ix = np.divide(Ix, g_mag)
    Iy = np.divide(Iy, g_mag)
    cres = 0
    prec = 0.4
    mSWT = -np.ones(i_img.shape)
    count = 1
    h_stroke = stroke_width * 0.5
    rows, cols = i_img.shape
    for i in range(rows):
        for j in range(cols):
            if (edgeImage[i, j] > 0):
                count = 0
                points_x = []
                points_y = []
                points_x.append(j)
                points_y.append(i)
                count += 1

                curX = float(j) + 0.5
                curY = float(i) + 0.5
                cres = 0
                while cres < stroke_width:
                    curX = curX + Ix[i, j] * prec  # find directionality increments x or y
                    curY = curY + Iy[i, j] * prec
                    cres = cres + 1
                    curPixX = int(math.floor(curX))
                    curPixY = int(math.floor(curY))
                    if (curPixX < 0 or curPixX > cols - 1 or curPixY < 0 or curPixY > rows - 1):
                        break
                    points_x.append(curPixX)
                    points_y.append(curPixY)
                    count += 1

                    if (edgeImage[curPixY, curPixX] > 0 and count < 21):
                        ang_plus = orientim[i, j] + angle
                        if (ang_plus > np.pi):
                            ang_plus = np.pi
                        ang_minus = orientim[i, j] - angle
                        if (ang_minus < 0):
                            ang_minus = 0
                        if ((orientim[curPixY, curPixX] < ang_plus) and (
                            orientim[curPixY, curPixX] > ang_minus) and count > h_stroke):
                            dist = math.sqrt((curPixX - j) * (curPixX - j) + (curPixY - i) * (curPixY - i))
                            for k in range(count - 1):
                                if (mSWT[points_y[k], points_x[k]] < 0):
                                    mSWT[points_y[k], points_x[k]] = dist
                                else:
                                    mSWT[points_y[k], points_x[k]] = np.min([dist, mSWT[points_y[k], points_x[k]]])
                        if (count > stroke_width):
                            break
    return mSWT

def SWT_Total(i_image, edges, orientation, stroke_width, angle=np.pi / 6):
    inv_iim = 255 - i_image  # needed for shadowing

    swtim = SWT(i_image, edges, orientation, stroke_width, angle)  # one image
    swtinv_im = SWT(inv_iim, edges, orientation, stroke_width, angle)  # the inverse

    swtim[np.nonzero(swtim < 0)] = 0
    swtinv_im[np.nonzero(swtinv_im < 0)] = 0
    swt_end = swtim
    indexes = np.nonzero(swtim == 0)
    swt_end[indexes] = swtinv_im[indexes]

    return swt_end

def hysthresh(image,T1,T2):
        if T1 < T2 :    # T1 and T2 reversed - swap values
	        tmp = T1
	        T1 = T2
	        T2 = tmp

        aboveT2 = image > T2;              # Edge points above lower threshold.
        [aboveT1r,aboveT1c] = np.nonzero(image > T1);  # Row and colum coords of points above upper threshold.
        # Obtain all connected regions in aboveT2 that include a point that has a
        # value above T1
        bw = floodfill(aboveT2, aboveT1r, aboveT1c, 8)
        return bw

def cleanswt2(swt,edges):
        mask = swt[swt > 0]
        labeled,nr_objects = label(mask)
        w, h = swt.shape
        max_pix = (0.05 * w)
        for i in range(nr_objects):
            numpix = len(np.where(labeled == i))
            if(numpix < max_pix):
                swt[np.where(labeled==i)] = 0
        swt[edges > 0] = np.max(swt)
        return swt

def autocanny(nm,canthresh):
        m,n = nm.shape
        im_size = np.array([m,n])

        med = float(np.median(nm[nm > 0]))
        max_factor = 0.95 * np.max(nm)
        factor_a = max_factor
        factor_b_p = 0.4*med

        bwedge = []
        value = 0
        msize =  m*n
        max_pix = int(msize*canthresh)
        iter = 0
        while (value < max_pix and iter<50):
            iter = iter+1
            bwedge = hysthresh(nm, factor_a * med, factor_b_p)
            value = np.sum(bwedge)
            factor_a = factor_a * 0.9
            if (factor_a < 1e-15):
                break

        c1 = 0
        alpha_1 = 0.01
        alpha_2 = 0.01
        inv = True
        iter = 0
        while (np.abs(value-max_pix)>200 and iter<20):
            bwedge = hysthresh(nm, factor_a * med, factor_b_p)
            value = np.sum(bwedge)
            iter = iter+1
            if(value<max_pix):
                if(inv):
                    alpha_1 = 0.01
                    inv = False
                factor_a = factor_a - alpha_1
                c1 = c1 + 1
                if(c1==2):
                    alpha_1 = alpha_1 * 2
                    c1 = 0
            else:
                if(not inv):
                    alpha_2 = 0.01
                    inv = True
                c1 = c1 - 1
                factor_a = factor_a + alpha_2
                if(c1 == -2 ):
                    alpha_2 = alpha_2 * 2
                    c1 = 0
        return bwedge

def autocanny2(prepro, nm, canthresh, blocksize):
        m,n = prepro.shape
        im_size = np.array([m,n])
        size_pixels = im_size / blocksize
        size_pixels = int(size_pixels[0] * size_pixels[1])

        # Clustering of image
        segments = slic(prepro, n_segments=size_pixels, sigma=1.5, compactness=0.08, start_label=0)
        num_labels = np.max(segments) + 1

        med = float(np.median(nm[nm > 0]))
        max_factor = 0.95 * np.max(nm)
        factor_a = max_factor
        factor_b_p = 0.4*med

        bwedge = []
        value = 0
        msize =  m*n
        max_pix = int(msize*canthresh)

        while (value < max_pix):
            bwedge = hysthresh(nm, factor_a * med, factor_b_p)
            value = np.sum(bwedge)
            factor_a = factor_a * 0.9
            if (factor_a < 1e-15):
                break

        f = []
        f.append(factor_a)
        factor_original = factor_a
        c1 = 0
        alpha_1 = 0.01
        alpha_2 = 0.01
        inv = True
        iter = 0
        while (np.abs(value-max_pix)>200 and iter<20):
            bwedge = hysthresh(nm, factor_a * med, factor_b_p)
            value = np.sum(bwedge)
            iter = iter+1
            if(value<max_pix):
                if(inv):
                    alpha_1 = 0.01
                    inv = False
                factor_a = factor_a - alpha_1
                c1 = c1 + 1
                if(c1==2):
                    alpha_1 = alpha_1 * 2
                    c1 = 0
            else:
                if(not inv):
                    alpha_2 = 0.01
                    inv = True
                c1 = c1 - 1
                factor_a = factor_a + alpha_2
                if(c1 == -2 ):
                    alpha_2 = alpha_2 * 2
                    c1 = 0
            f.append(factor_a)

        expected_density = (msize * canthresh) / size_pixels # Expected
        label_counter = 0
        for i in range(num_labels):
            label_density = np.sum(bwedge[np.where(segments == i)])
            if (label_density < 2 * expected_density):
                nm[segments == i]= 0
            else:
                bwedge[np.where(segments == i)] = 0;
                label_counter = label_counter + 1

        subsize = label_counter * blocksize * blocksize
        max_pix = (subsize/(msize*1.0))*canthresh
        factor_a = max_factor
        value = 0
        bwedge2 = np.zeros((m,n))
        while (value < max_pix):
            bwedge2 = hysthresh(nm, factor_a * med, factor_b_p);
            value = np.sum(bwedge2)/subsize
            factor_a = factor_a * 0.9;
            if (factor_a < 1e-15):
                break
        f = []
        f.append(factor_a)
        factor_original = factor_a
        c1 = 0
        alpha_1 = 0.01
        alpha_2 = 0.01
        inv = True
        iter = 0
        while (np.abs(value-max_pix)>0.001 and iter<20):
            bwedge2 = hysthresh(nm, factor_a * med, factor_b_p)
            value = np.sum(bwedge2)/subsize
            iter = iter+1
            if(value<max_pix):
                if(inv):
                    alpha_1 = 0.01
                    inv = False
                factor_a = factor_a - alpha_1
                c1 = c1 + 1
                if(c1==2):
                    alpha_1 = alpha_1 * 2
                    c1 = 0
            else:
                if(not inv):
                    alpha_2 = 0.01
                    inv = True
                c1 = c1 - 1
                factor_a = factor_a + alpha_2
                if(c1 == -2 ):
                    alpha_2 = alpha_2 * 2
                    c1 = 0
            f.append(factor_a)

        bwedge = np.logical_or(bwedge, bwedge2)
        return bwedge

def kuwahara_filter(input,winsize):
     # Kuwahara filters an image using the Kuwahara filter
     """
     filtered = Kuwahara(original, windowSize)
     filters the image with a given windowSize and yielsd the result in filtered
     It uses = variance = (mean of squares) - (square of mean).
     filtered = Kuwahara(original, 5);
     Description : The kuwahara filter workds on a window divide into 4 overlapping subwindows
     In each subwindow the mean and hte variance are computed. The output value (locate at the center of the window)
     is set to the mean of the subwindow with the smallest variance
     References:
     http: // www.ph.tn.tudelft.nl / DIPlib / docs / FIP.pdf
     http: // www.incx.nec.co.jp / imap - vision / library / wouter / kuwahara.html
     :param input:
     :param winsize:
     :return:
     """
     input = np.array(input,dtype = np.float64)
     m,n = input.shape
     if (winsize%4) != 1 :
        return

     tmpAvgKerRow = np.concatenate((np.ones( (1, (winsize - 1) / 2 + 1)), np.zeros((1, (winsize - 1) / 2))),axis=1)
     tmpPadder = np.zeros((1, winsize));
     tmpavgker = np.matlib.repmat(tmpAvgKerRow, (winsize - 1) / 2 + 1, 1)
     tmpavgker = np.concatenate((tmpavgker, np.matlib.repmat(tmpPadder, (winsize - 1) / 2, 1)))
     tmpavgker = tmpavgker / np.sum(tmpavgker)

     # tmpavgker is a 'north-west'
     t1,t2 = tmpavgker.shape
     avgker = np.zeros((t1,t2,4))
     avgker[:,:, 0] = tmpavgker # North - west(a)
     avgker[:,:, 1] = np.fliplr(tmpavgker) # North - east(b)
     avgker[:,:, 3] = np.flipud(tmpavgker) # South - east(c)
     avgker[:,:, 2] = np.fliplr(np.flipud(tmpavgker)) # South - west(d)

     squaredImg = input**2
     avgs = np.zeros((m,n,4))
     stddevs =  np.zeros((m,n,4))

     ## Calculation of averages and variances on subwindows
     for k in range(0,4):
        avgs[:,:, k] = convolve2d(input, avgker[:,:, k], 'same') # mean
        stddevs[:,:, k] = convolve2d(squaredImg, avgker[:,:, k], 'same') # mean
        stddevs[:,:, k] = stddevs[:,:, k]-avgs[:,:, k]**2 # variance

     # minima = np.min(stddevs, axis=2)
     indices = np.argmin(stddevs,axis = 2)
     filtered = np.zeros(input.shape)
     for k in range(m) :
        for i in range(n):
            filtered[k, i] = avgs[k, i, indices[k, i]]

     return filtered

def nonmaxsup_python(gradient,orientation,radius = 1.2):
        """
        # Input:
        #   inimage - Image to be non-maxima suppressed.

        #   orient  - Image containing feature normal orientation angles in degrees
        #             (0-180), angles positive anti-clockwise.
        #   radius  - Distance in pixel units to be looked at on each side of each
        #             pixel when determining whether it is a local maxima or not.
        #             This value cannot be less than 1.
        #             (Suggested value about 1.2 - 1.5)
        #    Returns:
        #   im        - Non maximally suppressed image.
        #
        # Notes:
        # The suggested radius value is 1.2 - 1.5 for the following reason. If the
        # radius parameter is set to 1 there is a chance that a maxima will not be
        # identified on a broad peak where adjacent pixels have the same value.  To
        # overcome this one typically uses a radius value of 1.2 to 1.5.  However
        # under these conditions there will be cases where two adjacent pixels will
        # both be marked as maxima.  Accordingly there is a final morphological
        # thinning step to correct this.

        # This function is slow.  It uses bilinear interpolation to estimate
        # intensity values at ideal, real-valued pixel locations on each side of
        #  pixels to determine if they are local maxima.

        # Copyright (c) 1996-2013 Peter Kovesi
        """
        im = np.zeros(gradient.shape)
        if(radius<1):
            return

        iradius = int(math.ceil(radius))

        # Precalculate x and y offsets relative to centre pixel for each orientation angle
        angle = range(0,181,1)
        angle = (np.array(angle)*np.pi)/180    # Array of angles in 1 degree increments (but in radians).
        xoff = radius*np.cos(angle)   # x and y offset of points at specified radius and angle
        yoff = radius*np.sin(angle)   # from each reference position.

        hfrac = xoff - np.floor(xoff) # Fractional offset of xoff relative to integer location
        vfrac = yoff - np.floor(yoff)    # Fractional offset of yoff relative to integer location

        orient = np.fix(orientation)    # Orientations start at 0 degrees but arrays start
                                        #  with index 1.
        orient = np.array(orient,dtype=np.int16)
        #  Now run through the image interpolating grey values on each side
        # of the centre pixel to be used for the non-maximal suppression.
        [rows,cols] = gradient.shape
        nrow = range(iradius+1,rows - iradius)
        ncol = range(iradius+1,cols - iradius)
        for elr in nrow:
            for elc in ncol:
                ori = orient[elr,elc]   # Index into precomputed arrays
                x = elc + xoff[ori]     # x, y location on one side of the point in question
                y = elr - yoff[ori]

                fx =  int(np.floor(x))        # Get integer pixel locations that surround location x,y
                cx =  int(np.ceil(x))
                fy =  int(np.floor(y))
                cy =  int(np.ceil(y))

                tl = gradient[fy,fx]    #  Value at top left integer pixel location.
                tr = gradient[fy,cx]    # top right
                bl = gradient[cy,fx]    # bottom left
                br = gradient[cy,cx]    # bottom right

                upperavg = tl + hfrac[ori]*(tr - tl)  # Now use bilinear interpolation to
                loweravg = bl + hfrac[ori]*(br - bl)  # estimate value at x,y
                v1 = upperavg + vfrac[ori]*(loweravg - upperavg)

                if (gradient[elr, elc] > v1): # We need to check the value on the other side...
                    x = elc - xoff[ori]       # x, y location on the `other side' of the point in question
                    y = elr + yoff[ori]

                    fx = int(np.floor(x))
                    cx = int(np.ceil(x))
                    fy = int(np.floor(y))
                    cy = int(np.ceil(y))
                    tl = gradient[fy,fx]   # % Value at top left integer pixel location.
                    tr = gradient[fy,cx]   # % top right
                    bl = gradient[cy,fx]   # % bottom left
                    br = gradient[cy,cx]   # % bottom right

                    upperavg = tl + hfrac[ori]*(tr - tl)
                    loweravg = bl + hfrac[ori]*(br - bl)
                    v2 = upperavg + vfrac[ori]*(loweravg - upperavg)

                    if (gradient[elr,elc] > v2):             # This is a local maximum.
                        im[elr, elc] = gradient[elr, elc]    # Record value in the output


        #  Finally thin the 'nonmaximally suppressed' image by pointwise
        #  multiplying itself with a morphological skeletonization of itself.
        #  I know it is oxymoronic to thin a nonmaximally supressed image but
        #  fixes the multiple adjacent peaks that can arise from using a radius
        #  value > 1.
        #
        # skel = bwmorph(im>0,'skel',Inf);
        #
        im2 = (im>0).astype(np.int8)
        skel= morphology.skeletonize(im2)
        im = np.multiply(im,skel)
        return im

def floodfill(bw, r, c, N=8):
    filled = np.zeros(bw.shape)
    theStack = deque(zip(r, c))

    while len(theStack) > 0:
        x, y = theStack.pop()
        if filled[x, y] == 1:
            continue
        if bw[x, y] == 0:
            continue
        filled[x, y] = 1
        theStack.append((x + 1, y))  # right
        theStack.append((x - 1, y))  # left
        theStack.append((x, y + 1))  # down
        theStack.append((x, y - 1))  # up
        if (N == 8):
            theStack.append((x + 1, y + 1))  # d right
            theStack.append((x - 1, y - 1))  # d left
            theStack.append((x - 1, y + 1))  # down
            theStack.append((x + 1, y - 1))  # up
    return filled

def borderEnhancer(img,filtersize):
    # Estimate the local mean of f.
    prod_fs  = reduce(lambda x, y: x * y, filtersize, 1)
    localMean = convolve2d(img,np.ones(filtersize),'same') / prod_fs;
    #  Estimate of the local variance of f.
    img_2 = np.multiply(img,img)
    localMean_2 = localMean*localMean
    localVar =  convolve2d(img_2,np.ones(filtersize),'same') / prod_fs - localMean_2;
    localVar = localVar>0
    return localVar

