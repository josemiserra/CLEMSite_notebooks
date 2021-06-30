
## CLEMSite a software for automated phenotypic screens using light microscopy and FIB-SEM

### Introduction

In Correlative Light and Electron Microscopy (CLEM), two imaging modalities are combined to take advantage of the localization capabilities of light microscopy to guide the capture of ultrastructure details in the electron microscope. In recent years, in the electron microscopy field, FIB/SEM (Focused Ion Beam Scanning Electron Microscope) tomography has emerged as a flexible method that enables semi-automated volume acquisition. With a FIB/SEM instrument large areas of a sample surface can be scanned with the electron beam to find regions of interest (ROIs) with high accuracy. Afterwards, combined with the FIB column, large 3D image stacks of biological samples are obtained with resolutions close to conventional TEM (Transmission Electron Microscopy). However, the process of identification and searching for regions of interest is a tedious manual task, prone to human errors and it prevents further scalability of the method. 


In the paper __CLEMSite a software for_automated phenotypic screens using light microscopy and FIB-SEM__ (https://www.biorxiv.org/content/10.1101/2021.03.19.436113v1) we introduced a complete automated workflow adapted to adherent cultured cells able to track and find cell regions previously identified in the light microscope. The workflow relies on the scanning capabilities of a FIB/SEM microscope coupled with a set of computational methods derived from computer vision, such as automated identification of landmarks and homography registration. The novel combination of these techniques enables the complete automation of the microscope.The software was tested in Hela cell culture samples coming from a phenotypic screening where cells were picked in the light microscope given specific properties of the Golgi apparatus. 

In this series of notebooks, the methods part is described in detail. In some cases is copying parts of the CLEMSite software functionality (available in https://github.com/josemiserra/CLEMSite), in others, they make available some code for reproducing the results of the paper. The main goal of these notebooks is to supplement the explanations offered in the paper, with the emphasis on making the tools available to the scientific community.


If you need to reference any of the notebooks in your work, use the paper citation: https://www.biorxiv.org/content/early/2021/03/19/2021.03.19.436113


### List of notebooks
-  **1  pattern_Recognition_LM_MatTek**
    In this notebook, in the first time we will obtain a set combinations of two letters and numbers and create a network to identify those characters.

-  **2  crossing_Detector_SEM_MatTek**
    This section contains 2 notebooks. During the first we train a network to detect edges of the grid and then we apply a line detector on them. In the second notebook, after applying the trained network and the line detector, the center of crossings is extracted and used as a landmark.
-  **3  terrain_Checker_MatTek**
    This notebook trains a network to recognize patches from pictures as inside/outside sample or damaged. 
-  **4  correlation_strategy_LM-SEM_global**
    This notebook uses the map system to calculate a global affine transform between landmarks in LM and EM after doing a scan of the grid. 
-  **5  correlation_strategy_LM-SEM_local**
    As continuation of the fourth notebook, a transform is applied now locally to a few landmarks. The purpose of this notebook is show a manual registration, and use it as a ground truth to compare it with the global results. 
-  **6 feedback_light_microscopy**
    These notebooks were used during the feedback microscopy workflow (between the prescan map and the high resolution acquisition), to select the cells showing the interesting phenotypes for CLEM.
    The cells of interest are selected using t-SNE or by selection of one particular feature.


### Getting Started

1. Clone the repo
   ```sh
   git clone https://github.com/josemiserra/CLEMSite_notebooks
   ```
2. If you don't have Anaconda or Miniconda installed, go to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda in your computer (miniconda is a lightweight version of the Anaconda python environment). 

3. It is recommended that you install your own environment with Conda. Create a **python 3.7.7** environment. 
    ``` sh
   conda create -n clemsite_nb python=3.7.7
   ```
    Follow the instructions here: [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After that, open an anaconda command prompt or a prompt, and activate your environment.
    ```sh
    activate clemsite_nb 
    ```
4. Move into the folder of the project and install the packages present in requirements.txt of the repository
   ```sh
   pip install -r requirements.txt
   ```
    If you experience problems when reading from requirements.txt, the installation of packages from scratch requires the following:

    ``` sh
    conda install jupyter notebook
    conda install tensorflow-gpu==1.15 # If no GPU = conda install tensorflow==1.15
    conda install keras
    conda install scikit-learn scikit-image
    conda install opencv
    conda install matplotlib seaborn
    conda install tqdm
    conda install pandas plotly
    conda install bokeh==2.0.2
    conda install holoviews==1.13.2
    pip install bitarray
    pip install imgaug
    ```


6. Inside your anaconda prompt, in the folder where you downloaded the repository, run jupyter notebook (More info in https://jupyter.org/).
   ```sh
   jupyter notebook
   ```
   Run the `.ipynb`. You can then execute the cells pressing one by one by simply follow the notebook. If you never used a jupyter notebook, you will find plenty of tutorials on internet, (e.g. https://realpython.com/jupyter-notebook-introduction/). 

7. In some notebooks is necessary to download and extract some data, made available in Google Drive. In the notebook folder you will find a text file "download_data.txt". Download the data from the link, create a folder called __data__ and unzip it there. 



## License

Distributed under the MIT License. See `LICENSE` file for more information.


## Contact

You can contact the author directly or create an issue request in https://github.com/josemiserra/CLEMSite_notebooks/issues

Jose Miguel Serra Lleti - serrajosemi@gmail.com

