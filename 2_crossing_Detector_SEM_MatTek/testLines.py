import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time 
from shutil import copyfile
from tqdm import tqdm
import glob
import os
from keras import backend as K
import os
from keras.preprocessing import image                  
from skimage import data, io, filters
from keras.utils import np_utils
from sklearn.datasets import load_files 
from sklearn.model_selection import train_test_split
from skimage.morphology import square
from skimage.filters.rank import mean_bilateral
from skimage.morphology import erosion, dilation
from importlib import reload
import numpy as np
import cv2

# load json and create model
from keras.models import model_from_json

iclahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(32, 32))

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

# All images are preprocessed for unbalanced brightness and contrast and resized to 256 by 256
def preprocess(image):
    image = cv2.GaussianBlur(image,(5,5),0)
    n_image = iclahe.apply(np.uint8(image))
    final = np.array(n_image,dtype=np.float32)
    final = (final-np.min(final))/(np.max(final)-np.min(final)) # Normalization has to be done AFTER AUGMENTATION
    return final

def path_to_tensor(img_path):
    img = cv2.imread(img_path,0)
    final = cv2.resize(img,(256,256))
    final = preprocess(final)
    # convert 3D tensor to 4D tensor with shape (1, 128, 128, 1) and return 4D tensor
    return np.expand_dims(final, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)




def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)






set_keras_backend('tensorflow')

# Load TEST images
sample_test = []
# Load data
dirSample = ".\\data\\crosses"
flist = glob.glob(dirSample + "\\*TEST*\\f*")
flistdir= [f for f in flist if os.path.isdir(f)]
if flistdir is None:
        raise Exception('--scanfolder not found')

for sample in flistdir:
    flist2 = glob.glob(sample + "\\ref*.tif")
    sample_test.append(flist2[0])

test_tensors = paths_to_tensor(sample_test)
test_tensors = np.expand_dims(test_tensors, axis=3)

json_file = open('model_CROSS_DETECTOR.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# loaded_model.load_weights('weights.bestM_UNET.from_scratch.hdf5')
loaded_model.load_weights('weights.model_CROSS_DETECTOR.hdf5')
print("Loaded model from disk")

y_test = loaded_model.predict(test_tensors)

# fig = plt.figure(figsize=(20,40))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# for i in range(len(sample_test)):
#    ax = fig.add_subplot(9, 6, i + 1, xticks=[], yticks=[])
#    img = np.zeros((256,256,3),dtype = np.uint8)
#    img[:,:,:]= test_tensors[i]*255
#    im1 = img[:,:,0]
#    im1[np.squeeze(y_test[i,:,:]>0.15)] = 255
#    img[:,:,0] = im1
#    ax.imshow(np.squeeze(img))

# plt.pause(1) # <-------
# plt.waitforbuttonpress(0)
# plt.close(fig)

from cross_detector_utils import *
# TEST calculate orientations
img = test_tensors[16,:,:]
img = img.reshape((256,256))
orientations = calculateOrientations(img[:,:])
# plt.imshow(orientations)
# plt.waitforbuttonpress(0)
# plt.close(fig)
img_p =y_test[16,:,:]
img_p = img_p.reshape((256,256))


# prj_pos = projections(img_p>0.5,orientations,aspace=[45], arange = 10)
# prj_neg = projections(img_p>0.5,orientations,aspace=[-45], arange = 10)
# plt.figure(figsize=(10, 7))
# plt.imshow(prj_neg+prj_pos,'hot')
# plt.colorbar()

# peaks_pos, hnew = detectPeaksNMS(prj_pos, numpeaks = 6,threshold = 0.1, nhood = None)
# peaks_neg, hnew = detectPeaksNMS(prj_neg, numpeaks = 6,threshold = 0.1, nhood = None)
# print(peaks_pos)
# res = findCrossSequence(peaks_pos,[35])
# print(res)
# signal = prj_pos[:,133].copy() # get complementary data
# plt.plot(signal)

# speaks = nonmaxsup1D(signal,6,int(np.array([35])*0.5),0.01)
# print(speaks)





# orientations = calculateOrientations(img)
# prj_pos = projections(img_p>0.6,orientations,aspace=[45], arange = 10)
# prj_neg = projections(img_p>0.6,orientations,aspace=[-45], arange = 10)
# R = prj_pos+prj_neg

# peaks_pos, hnew = detectPeaksNMS(prj_pos, numpeaks = 6,threshold = 0.1, nhood = None)
# peaks_neg, hnew = detectPeaksNMS(prj_neg, numpeaks = 6,threshold = 0.1, nhood = None)
# print(peaks_pos)
# print(peaks_neg)
# discardwrongpeaks(R,peaks_pos,peaks_neg,[35])
imgw, fa = autoThreshold(img_p,0.1)
fpeaks_pos, fpeaks_neg = getPeaks(img,imgw,[35])

if fpeaks_pos.shape[0] == 0 or  fpeaks_neg.shape[0] == 0 :
    fpeaks_pos, fpeaks_neg = getPeaks(img, imgw, [18])
    if fpeaks_pos.shape[0] == 0 or fpeaks_neg.shape[0] == 0:
        print("No peaks found")
print(fpeaks_pos)
print(fpeaks_neg)
cpoint, fpoints, iimgc, imgl =selectGridPoints(np.uint8(img*255),fpeaks_pos, fpeaks_neg)
print(cpoint)