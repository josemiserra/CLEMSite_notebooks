
import os
from os import listdir
from os.path import isfile, join
import cv2
from skimage import filters
import numpy as np

DIGITS_FN = os.getcwd()+'\\codebook'
DIGITS_FN2 = os.getcwd()+'\\Synthetic2'

def load_digits(mypath):
        ofile = {}
        tag = []
        onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
        for el in onlyfiles:
            tag.append(el[0:2]); #label
            ofile[el[0:2]] = mypath+'\\'+el;
        return ofile,tag


def gaussianNoise(image):
    row,col= image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

def salt_n_pepper(image):
    row,col = image.shape
    s_vs_p = np.random.uniform(0, 1)
    amount = np.random.uniform(0.02, 0.2)
    out = image
      # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    #Now, for each coordinate, generate a small random crap
    out[coords] = 255

    # Pepper mode
    amount = np.random.uniform(0.02, 0.2)
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out

def poisson(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def speckle(image):
    row,col = image.shape
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)
    noisy = image + image * gauss
    return noisy


def destroy_n_crap(image,bw):
    row,col = image.shape
    s_vs_p = np.random.uniform(0, 1)
    amount = np.random.uniform(0.02, 0.5)
    out = image
      # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    for el in coords:
            if(len(el)<2):
                continue
            snake_len = np.random.poisson(30,1)
            inc = np.random.poisson(4,8)
            new_x = el[0]
            new_y = el[1]
            for step in range(0,snake_len):
                inc = np.random.poisson(4)
                while(inc>7):
                    inc = np.random.poisson(4)
                if(inc==0):
                    new_x = new_x+inc
                    new_y = new_y
                elif(inc==1):
                    new_x = new_x+inc
                    new_y = new_y+inc
                elif(inc==2):
                    new_x = new_x+inc
                    new_y = new_y-inc
                elif(inc==3):
                    new_x = new_x-inc
                    new_y = new_y-inc
                elif(inc==4):
                    new_x = new_x
                    new_y = new_y+inc
                elif(inc==5):
                    new_x = new_x
                    new_y = new_y-inc
                elif(inc==6):
                    new_x = new_x-inc
                    new_y = new_y
                elif(inc==7):
                    new_x = new_x-inc
                    new_y = new_y+inc
                if(new_x<col and new_y<row and new_x>-1 and new_y>-1):
                    out[new_x][new_y] = bw
                inc2 = np.random.randint(1,10)
                if(new_x+inc2<col and new_y+inc2<row and new_x-inc2>-1 and new_y-inc2>-1):
                    out[new_x+inc2][new_y] = bw
                    out[new_x][new_y+inc2] = bw
                    out[new_x-inc2][new_y] = bw
                    out[new_x][new_y-inc2] = bw
                    out[new_x+inc2][new_y+inc2] = bw
                    out[new_x-inc2][new_y-inc2] = bw
                    out[new_x-inc2][new_y+inc2] = bw
                    out[new_x+inc2][new_y-inc2] = bw
    return out


def random_spots(image,bw):
    n_circles = np.random.randint(1,3)
    rows, cols = image.shape
    for i in range(0,n_circles):
        c_x = np.random.randint(0, cols)
        c_y = np.random.randint(0, rows)
        radius = np.random.poisson(2)
        cv2.circle(image, (c_x,c_y), radius,(bw,bw,bw),5)

    n_ellipses = np.random.randint(1,4)
    for i in range(0,n_ellipses):
        c_x = np.random.randint(0, cols)
        c_y = np.random.randint(0, rows)
        ax1 = np.random.poisson(3)
        ax2 = np.random.poisson(10)
        angle = np.random.randint(0,180)
        cv2.ellipse(image, (c_x,c_y),(ax1,ax2),angle,0,360,(bw,bw,bw),10)
    n_ellipses = np.random.randint(1,4)
    for i in range(0,n_ellipses):
        c_x = np.random.randint(0, cols)
        c_y = np.random.randint(0, rows)
        ax1 = np.random.poisson(8)
        ax2 = np.random.poisson(3)
        angle = np.random.randint(0,181)
        cv2.ellipse(image, (c_x,c_y),(ax1,ax2),angle,0,360,(bw,bw,bw),10)
    return image


def generate_snakes(image,bw):
    row,col = image.shape
    s_vs_p = np.random.uniform(0.2, 1)
    amount = np.random.uniform(0.02, 0.5)
    out = image
      # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    #Now, for each coordinate, generate a small random crap

    # for each coord add elements
    for el in coords:
            inc = np.random.poisson(15,8)
            new_x = np.array(range(el[0],el[0]+inc[0]))
            new_y = np.array(new_x,copy=True)
            new_y.fill(el[1])
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                  out[el1[0],el1[1]] =bw
                  last_x = el1[0]
                  last_y = el1[1]
            m_continue = np.random.randint(0,2)
            if(m_continue==1):
                el[0]=last_x
                el[1]=last_y
            new_x = np.array(range(el[0]-inc[1],el[0]))
            new_y = np.array(new_x,copy=True)
            new_y.fill(el[1])
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                    out[el1[0],el1[1]] =bw
                    last_x = el1[0]
                    last_y = el1[1]
            m_continue = np.random.randint(0,2)
            if(m_continue==1):
                el[0]=last_x
                el[1]=last_y
            new_y = np.array(range(el[1]-inc[2],el[1]))
            new_x = np.array(new_x,copy=True)
            new_x.fill(el[0])
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                    out[el1[0],el1[1]] =bw
                    last_x = el1[0]
                    last_y = el1[1]
            m_continue = np.random.randint(0,2)
            if(m_continue==1):
                el[0]=last_x
                el[1]=last_y
            new_y = np.array(range(el[1],el[1]+inc[3]))
            new_x = np.array(new_x,copy=True)
            new_x.fill(el[0])
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                    out[el1[0],el1[1]] =bw
                    last_x = el1[0]
                    last_y = el1[1]
            m_continue = np.random.randint(0,2)
            if(m_continue==1):
                el[0]=last_x
                el[1]=last_y
            new_y = np.array(range(el[1]-inc[4],el[1]))
            new_x = np.array(range(el[0],el[0]+inc[4]))
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                    out[el1[0],el1[1]] =bw
                    last_x = el1[0]
                    last_y = el1[1]
            m_continue = np.random.randint(0,2)
            if(m_continue==1):
                el[0]=last_x
                el[1]=last_y
            new_y = np.array(range(el[1]-inc[5],el[1]))
            new_x = np.array(range(el[0]-inc[5],el[0]))
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                    out[el1[0],el1[1]] =bw
                    last_x = el1[0]
                    last_y = el1[1]
            m_continue = np.random.randint(0,2)
            if(m_continue==1):
                el[0]=last_x
                el[1]=last_y
            new_y = np.array(range(el[1],el[1]+inc[6]))
            new_x = np.array(range(el[0],el[0]+inc[6]))
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                    out[el1[0],el1[1]] =bw
                    last_x = el1[0]
                    last_y = el1[1]
            m_continue = np.random.randint(0,2)
            if(m_continue==1):
                el[0]=last_x
                el[1]=last_y
            new_y = np.array(range(el[1],el[1]+inc[7]))
            new_x = np.array(range(el[0]-+inc[7],el[0]))
            for el1 in zip(new_x,new_y):
                if(el1[0]<256 and el1[1]<256 and el1[0]>-1 and el1[1]>-1):
                    out[el1[0],el1[1]] =bw
    return out

if __name__ == '__main__':
    print(__doc__)
    digits, labels = load_digits(DIGITS_FN)
    print(labels)
    # Let's play. The idea is generate a dataset where for each
    # letter we have randomly, missing edges, noise and chunks
    # out of place of different size
    count = 0
    for el in labels:
        adir = DIGITS_FN2+"\\"+labels[count]
        if not os.path.exists(adir):
            os.makedirs(adir)
        for i in range(0,20):
             my_file_dig = digits[el]
             my_digit = cv2.imread(my_file_dig, cv2.IMREAD_GRAYSCALE)
             rows,cols = my_digit.shape
             img = (my_digit>128)*255 # invert
             my_digit = filters.sobel(img)
             # Apply random dilation (1-3)
             k = np.random.randint(0, 7)
             my_digit = np.uint8(my_digit>0)
             if(k>0):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                my_digit = cv2.dilate(my_digit, kernel)
             # Apply random rotation
             rangle = np.random.randint(-5,5)
             M = cv2.getRotationMatrix2D((cols/2,rows/2),rangle,1)
             my_digit = cv2.warpAffine(my_digit,M,(cols,rows))

             r1 = np.random.randint(-15,15,size=2)
             M = np.float32([[1,0,r1[0]],[0,1,r1[1]]])
             my_digit = cv2.warpAffine(my_digit,M,(cols,rows))
             m_continue = np.random.randint(0,2)
             if(m_continue==1):
                random_spots(my_digit,255)
                random_spots(my_digit,0)
             m_continue = np.random.randint(0,2)
             if(m_continue==1):
                random_spots(my_digit,0)
             m_continue = np.random.randint(0,2)
             if(m_continue==1):
                my_digit= destroy_n_crap(my_digit,255)
             m_continue = np.random.randint(0,2)
             if(m_continue==1):
                my_digit = destroy_n_crap(my_digit,255)
             m_continue = np.random.randint(0,2)
             if(m_continue==1):
                my_digit = destroy_n_crap(my_digit,0)
             m_continue = np.random.randint(0,2)
             if(m_continue==1):
                my_digit = destroy_n_crap(my_digit,0)
             m_continue = np.random.randint(0,2)
             if(m_continue==1):
                my_digit = generate_snakes(my_digit,255)
                my_digit = generate_snakes(my_digit,255)
             myim_name = labels[count]+"_"+str(i)+".tif"
             mydir = adir+"\\"+myim_name
             cv2.imwrite(mydir,(my_digit>0)*255)
        count = count+1








