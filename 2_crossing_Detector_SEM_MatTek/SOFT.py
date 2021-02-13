
import sys, os
import time
import cv2
import numpy as np
import math
from scipy.signal  import convolve2d
from scipy.ndimage import label,sum
from scipy.misc import imrotate
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.segmentation import slic
from bwmorph import bwmorph_thin
from collections import deque
class SOFT:


 #   def __init__(self):
 #       pass

    def soft(self,image,sigma=1.5, clahe=True, canny_thresh=0.05, stroke=20):
        """
            Find Lines using Orientations and Projections
                        Note: +CT means, increases computation time.
            'K'            Number of neighbors to consider in the Orientation Field
                      Transform.Each neighbor is evaluated against a candidate angle
                      and then add up. The biggest the lines, the better the
                      result for a bigger K. (K big,+CT)
                      Default: 12.
            'Delta'        Angle increment from 0.1 to 90.
                      The resulting projection will be more accurate if
                      the increment is small. (Delta small, +CT)
                      Default: 1.
            'dispim'       If true, images are shown. If false,no images are shown.
                      Default: True
                      %
            'wiener'       Two-element vector of positive integers: [M N].
                      [M N] specifies the number of tile rows and
                     columns.  Both M and N must be at least 2.
                      The total number of image tiles is equal to M*N.
                      If the lines are too thin, it is used to dilate them.
                      Default: [2 2].
                      Use 0 to not execute the wiener filter.

            'strokeWidth'  When the Stroke Width Transform is executed, for each
                      pixel, rays are created from the pixel to the next
                      change of gradient. If your stroke is big, use a bigger
                      width.(strokeWidth big, ++CT)
                      Default: 20.
            'canthresh'    Automatic canny thresholding is performed using an
                      iterative loop. If the percentage of white pixels is bigger than a
                      threshold,then we are assuming the image is getting more
                      and more clutter.
                      Default: 0.075, means a 7.5% of the pixels is white.
            'Sigma'        Preprocessing gaussian filter. Helps with noisy images
                      of after CLAHE. Values between 0 and 2 are recommended.

                      Default: 0 (not applied).
            'clahe'        If true, CLAHE (automatic brightness and contrast
                      balance) is applied.
                      Default: False.
            ##########################################
            saved values:
            R  - map of projections
            peaks - peaks detected
            prepro - image after preprocessing (clahe and gaussian filter)
            bwedge - image after automatic canny filtering
            orientim - image with ridge orientations
            reliability - probabilistic plot of orientations
        :return:
        """
        if (sigma < 0):
            print 'Invalid value. Sigma cannot be smaller than 0.'
            sigma = 0

        self.sigma = sigma
        self.clahe = clahe
        if (canny_thresh > 1 or canny_thresh <= 0):
            print 'Invalid threshold. Cannot be bigger than 1 or smaller than 0. Setting default value.'
            canny_thresh = 0.05
        self.canthresh = canny_thresh


        if (stroke < 2 or stroke > 1024):
            print 'Invalid stroke size. Accepted values between and half the size of your image.Setting default value.'
            stroke = 20
        self.stroke = stroke


        print("Preprocessing")
        start = time.time()
        prepro = image
        if(self.clahe):

            print('CLAHE true, performed at clipLimit 0.01 and tileGridSize of 32,32')

        if(self.sigma>0):
            sze = int(math.ceil(6*self.sigma))
            if(sze%2 == 0):
                sze = sze+1
            h = self.fspecial_gauss2D((sze,sze),self.sigma)
            I = convolve2d(prepro,h,'same')
            print('Gaussian blur performed with Size ' +str(sze)+ ' and sigma '+ str(self.sigma))

        PREPRO = I

        end = time.time()
        print "Preprocessing done: "+str(end - start)+" s."
        ##### Gradient
        start = time.time()
        gradient,orientation = self.canny(I,2)
        end = time.time()
        print "Gradient done: "+str(end - start)+" s."
       # plt.subplot(121),plt.imshow(orientation,cmap='gray')
       # plt.subplot(122),plt.imshow(gradient*10,cmap='gray')
       # plt.show()

        start = time.time()
    #   nm =  self.nonmaxsup(gradient,orientation,1.5)
        nm =  self.nonmaxsup_python(gradient,orientation,1.5)
        end = time.time()
        print "NMS done: "+str(end - start)+" s."


        start = time.time()
     #  nm =  nonmaxsup(gradient,orientation,1.5)
        BWEDGE =  self.autocanny2(prepro,nm,16)
        end = time.time()
        print "Autocanny done: "+str(end - start)+" s."


        m_size = np.array([2,2])
        J = self.borderEnhancer(BWEDGE,m_size)
        print 'Border Enhancement done'
        start = time.time()
        ORIENTIM, _reliability = self.ridgeorient(gradient, 1, 5, 5)
        segments = slic(prepro, n_segments=2500, sigma=1.5, compactness=0.08)
        num_labels = np.max(segments) + 1
        orientim_slic = np.copy(ORIENTIM)
        for i in range(num_labels):
            orientim_slic[np.where(segments == i)] = np.median(ORIENTIM[np.where(segments == i)])
        ORIENTIM = orientim_slic

        _, RELIABILITY = self.ridgeorient(gradient, 1, 3, 3)
        RELIABILITY[RELIABILITY<0.5] = 0
        end = time.time()
        print "Ridges done: "+str(end - start)+" s."
        # plt.imshow(orientim2 ,cmap='jet')
        tl = np.multiply(J,RELIABILITY)  # Enhance the bw image removing disordered regions

        if self.stroke>0:
            print "Starting SWT with strokeWidth of "+str(self.stroke)

            start = time.time()
            iSWT= self.SWT_Total(I,tl,ORIENTIM,self.stroke)
            end = time.time()
            print "SWT done: "+str(end - start)+" s."

            start = time.time()
            print('Removing ill components')
            FSWT = self.cleanswt2(iSWT,J)
            end = time.time()
            print "Removing done: " + str(end - start) + " s.\n"

        plt.show()
        return PREPRO,BWEDGE,ORIENTIM,RELIABILITY,iSWT,FSWT

    def autocanny(self,nm):
        med = np.median(nm[nm>0])
        max_factor = 0.8*np.max(nm)
        factor_a =  max_factor
        factor_b = 0.4


        lenm = nm.shape
        bwedge = np.zeros(lenm)
        value = 0
        msize = (lenm[0]*lenm[1])
        while(value<self.canthresh):
            bwedge = self.hysthresh(nm, factor_a*med,factor_b*med);
            value = np.sum(bwedge)/msize
            factor_a = factor_a*0.9

        # Coarse part or histeresis accomplished
        while(value>self.canthresh):
            factor_a = factor_a + 0.01
            bwedge = self.hysthresh(nm, factor_a*med,factor_b*med);
            value = np.sum(bwedge)/msize

        print 'Automatic Canny Done'
        print 'Lower threshold reached at '+str(factor_b)
        print 'Upper threshold reached at '+str(factor_a)
        return bwedge

    def autocanny2(self, prepro, nm, blocksize):
        m,n = prepro.shape
        im_size = np.array([m,n])
        size_pixels = im_size / blocksize
        size_pixels = int(size_pixels[0] * size_pixels[1])

        # Clustering of image
        segments = slic(prepro, n_segments=size_pixels, sigma=1.5, compactness=0.08)
        num_labels = np.max(segments) + 1

        med = float(np.median(nm[nm > 0]))
        max_factor = 0.95 * np.max(nm)
        factor_a = max_factor
        factor_b = 0.4

        bwedge = []
        value = 0
        msize =  m*n
        while (value < self.canthresh):
            bwedge = self.hysthresh(nm, factor_a * med, factor_b * med)
            value = np.sum(bwedge)/msize
            factor_a = factor_a * 0.9
            if (factor_a < 1e-15):
                break
        while (value > self.canthresh):
            factor_a = factor_a + 0.01
            bwedge = self.hysthresh(nm, factor_a * med, factor_b * med);
            value = np.sum(bwedge)/msize

        expected_density = (msize * self.canthresh) / size_pixels # Expected
        label_counter = 0
        for i in range(num_labels):
            label_density = np.sum(bwedge[np.where(segments == i)])
            if (label_density < 2 * expected_density):
                nm[segments == i]= 0
            else:
                bwedge[np.where(segments == i)] = 0;
                label_counter = label_counter + 1

        subsize = label_counter * blocksize * blocksize
        canthresh = (subsize/(msize*1.0))*self.canthresh
        factor_a = max_factor
        factor_b = 0.4
        value = 0
        bwedge2 = np.zeros((m,n))
        while (value < canthresh):
            bwedge2 = self.hysthresh(nm, factor_a * med, factor_b * med);
            value = np.sum(bwedge2) / subsize;
            factor_a = factor_a * 0.9;
            if (factor_a < 1e-15):
                break
        while (value > canthresh):
            factor_a = factor_a + 0.01;
            bwedge2 = self.hysthresh(nm, factor_a * med, factor_b * med);
            value = sum(sum(bwedge2)) / msize

        bwedge[bwedge2>0] = 1
        print 'Automatic Canny Done'
        print 'Lower threshold reached at ' + str(factor_b)
        print 'Upper threshold reached at ' + str(factor_a)
        return bwedge

    def gaussfilt(self,img,sigma):
        sze = int(math.ceil(6*sigma))
        if(sze%2 == 0):
            sze = sze+1
        h = self.fspecial_gauss2D((sze,sze),sigma)
        # conv2(image, mask) is the same as filter2(rot90(mask,2), image)
        image = convolve2d(img,h,'same')
        return image

    def derivative5(self,i_image):
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

    def fspecial_gauss2D(self,shape=(3,3),sigma=0.5):
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

    def nonmaxsup_python(self,gradient,orientation,radius = 1.2):
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
            print 'ERROR: radius should be bigger than 1'
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

    def hysthresh(self,image,T1,T2):
        if T1 < T2 :    # T1 and T2 reversed - swap values
	        tmp = T1
	        T1 = T2
	        T2 = tmp

        aboveT2 = image > T2;              # Edge points above lower threshold.
        [aboveT1r,aboveT1c] = np.nonzero(image > T1);  # Row and colum coords of points above upper threshold.
        # Obtain all connected regions in aboveT2 that include a point that has a
        # value above T1
        bw = self.floodfill(aboveT2, aboveT1r, aboveT1c, 8)
        return bw

    def floodfill(self,bw, r, c, N=8):
        filled = np.zeros(bw.shape)
        theStack = deque(zip(r,c))

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

    def borderEnhancer(self,img,filtersize):
        # Estimate the local mean of f.
        prod_fs  = reduce(lambda x, y: x * y, filtersize, 1)
        localMean = convolve2d(img,np.ones(filtersize),'same') / prod_fs;
        #  Estimate of the local variance of f.
        img_2 = np.multiply(img,img)
        localMean_2 = localMean*localMean
        localVar =  convolve2d(img_2,np.ones(filtersize),'same') / prod_fs - localMean_2;
        localVar = localVar>0
        return localVar

    def ridgeorient(self,im,gradientsigma,blocksigma,orientsmoothsigma):
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
        h = self.fspecial_gauss2D((sze,sze),gradientsigma)
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
        h = self.fspecial_gauss2D((sze,sze),blocksigma)
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
        h = self.fspecial_gauss2D((sze,sze),orientsmoothsigma)

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
        reliability = np.multiply(reliability,(denom>.001))
        return orientim,reliability

    def SWT(self,i_img,edgeImage,orientim,stroke_width=20,angle=np.pi/6):
        im = self.gaussfilt(i_img,1)
        Ix,Iy = self.derivative5(im)
        Ix_2 = np.multiply(Ix,Ix)
        Iy_2 = np.multiply(Iy,Iy)
        g_mag = np.sqrt(Ix_2 + Iy_2)   # Gradient magnitude.
        Ix = np.divide(Ix,g_mag)
        Iy = np.divide(Iy,g_mag)
        cres = 0
        prec = 0.4
        mSWT = -np.ones(i_img.shape)
        count = 1
        h_stroke = stroke_width*0.5
        rows,cols = i_img.shape
        for i in range(rows):
              for j in range(cols):
                    if(edgeImage[i,j]>0):
                        count = 0
                        points_x = []
                        points_y = []
                        points_x.append(j)
                        points_y.append(i)
                        count += 1

                        curX = float(j)+0.5
                        curY = float(i)+0.5
                        cres = 0
                        while cres<stroke_width :
                            curX = curX + Ix[i,j]*prec # find directionality increments x or y
                            curY = curY + Iy[i,j]*prec
                            cres = cres +1
                            curPixX =  int(math.floor(curX))
                            curPixY =  int(math.floor(curY))
                            if(curPixX<0 or curPixX > cols-1 or curPixY<0 or curPixY>rows-1):
                               break
                            points_x.append(curPixX)
                            points_y.append(curPixY)
                            count +=1

                            if(edgeImage[curPixY,curPixX]>0 and count<21):
                                ang_plus =  orientim[i,j]+angle
                                if(ang_plus>np.pi):
                                    ang_plus = np.pi
                                ang_minus = orientim[i,j]- angle
                                if(ang_minus<0):
                                    ang_minus = 0
                                if((orientim[curPixY,curPixX]<ang_plus) and (orientim[curPixY,curPixX]>ang_minus) and count> h_stroke ):
                                    dist= math.sqrt((curPixX - j)*(curPixX - j) + (curPixY-i)*(curPixY-i))
                                    for k in range(count-1):
                                        if(mSWT[points_y[k],points_x[k]]<0):
                                            mSWT[points_y[k],points_x[k]]=dist
                                        else:
                                            mSWT[points_y[k],points_x[k]]= np.min([dist,mSWT[points_y[k],points_x[k]]])
                                if(count>stroke_width):
                                    break
        return mSWT

    def SWT_Total(self,i_image,edges,orientation,stroke_width, angle = np.pi/6):

         inv_iim = 255 - i_image # needed for shadowing

         swtim =     self.SWT(i_image,edges,orientation,stroke_width,angle)  # one image
         swtinv_im = self.SWT(inv_iim,edges,orientation,stroke_width,angle)  # the inverse



         swtim[np.nonzero(swtim<0)]=0
         swtinv_im[np.nonzero(swtinv_im<0)]=0
         swt_end = swtim
         indexes = np.nonzero(swtim==0)
         swt_end[indexes] =  swtinv_im[indexes]

         return swt_end

    def cleanswt(self,image,edges):
         # find connected components
            labeled, nr_objects = label(image > 0)
            print "Number of objects is "+str(nr_objects)
            # image = binary_opening(image>0, structure=np.ones((3,3))).astype(np.int)
            mask = image > image.mean()
            sizes = sum(mask, labeled, range(nr_objects + 1))
            mask_size = sizes < 0.05*image.shape[0]
            remove_pixel = mask_size[labeled]
            image[remove_pixel]=0
            edges[edges>0] = np.max(image)
            return image+edges

    def cleanswt2(self,swt,edges):
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

    def projections(self,iswt, iorient, K, inc, aspace = False, arange = None):
        if (K < 4 or K > 1024):
            print 'Invalid average value. Accepted values between 4 and half the size of your image. Setting default value.'
            K = 12

        if (inc > 90 or inc < 0):
            print 'Invalid Delta, must be positive and less than 90'
            inc = 1
        print "Starting projections"
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
        THETA = range(-90,90,inc)
        th = np.zeros(len(THETA))+np.inf
        if(arange):
            for ang in aspace:
                 k = ang+90
                 kplus = k+range
                 kminus = k-range
                 if(kplus>180):
                     kplus = 180
                 if(kminus<0):
                     kminus = 1
                 th[range(k,kplus)] = THETA[range(k,kplus)]
                 th[range(kminus,k)] = THETA[range(kminus,k)]
        else:
            th = THETA
        th = np.array(th,dtype =np.float32)*np.pi*(1/180.0)

        n = len(THETA)
        PR = np.zeros((padIMG.shape[1], n))
        M = padIMG > 0

        iPL,iPW = padIMG.shape
        center = (iPL / 2, iPW / 2)
        for i in range(n):
            if(th[i]!=np.inf):
                final = self.oft(M,K, padIMGOR,th[i])
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

    def oft(self,M,K,L,ang):
        kernel = np.zeros((K,K))
        v_cos = math.cos(ang)
        v_sin = math.sin(ang)
        Mval = np.cos(2*(L-ang))

        count = 0
        for k in range(-K/2-1,K/2+2):
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

    def canny(self,i_image, isigma):
        image = self.gaussfilt(i_image, isigma)
        Ix, Iy = self.derivative5(image)
        Ix_2 = np.multiply(Ix, Ix)
        Iy_2 = np.multiply(Iy, Iy)
        gradient = np.sqrt(Ix_2 + Iy_2)  # Gradient magnitude.
        orientation = np.arctan2(-Iy, Ix)  # Angles -pi to + pi.
        orientation[orientation < 0] = orientation[orientation < 0] + np.pi;  # Map angles to 0-pi.
        orientation = orientation * 180 / np.pi;
        return gradient, orientation