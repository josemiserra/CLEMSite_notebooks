#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# AUTHOR: Jose Miguel Serra Lleti, EMBL 
# converter.py 
# To further usage, use manual supplied together with CLEMSite

import numpy as np
from sklearn.manifold import TSNE
import pylab as Plot
import pandas as pd
import glob
from os import listdir
from os.path import isfile, join
import os, shutil
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as mpatches
import holoviews as hv
from holoviews import streams
hv.extension('bokeh')
import cv2
import time
from skimage import data, exposure, img_as_float


############################################## LOADING DATA ############################################
### Set here treatment names
_treatments = ["COPB2","WDR75","DNM1","COPG1","C1S","DENND4C","IPO8","SRSF1","Neg9","FAM177B","ACTR3","PTBP1","DNM1","NT5C","PTBP1","ARHGAP44","Neg9","ACTR3","SRSF1","C1S","IPO8","WDR75","NT5C","FAM177B","COPB1","ARHGAP44","Neg9","GPT","KIF11","GPT","DENND4C","AURKB"]
_features =  ['Metadata_BaseFileName','FileName_ColorImage','Location_Center_X','Location_Center_Y','Mean_Golgi_AreaShape_Center_X','Mean_Golgi_AreaShape_Center_Y','Mean_Nuclei_AreaShape_Solidity','Metadata_U','Metadata_V','Metadata_X','Metadata_Y', 'ImageQuality_PowerLogLogSlope_Dna',
     'Intensity_IntegratedIntensity_GolgiBGCorr', 'Mean_Nuclei_Math_CV', 'Math_Blobness', 'Math_Diffuseness', 'Children_Golgi_Count', 'Mean_MaxAreaGolgi_AreaShape_FormFactor']

treatment_column = 'Gene'	 
treatment_index = 'Metadata_Y'
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
########################################################################################################

def loadData(regenerate_crops = True,no_treatments=False):
	"""
		regenerate_crops = rewrites crops. Put it to false to save computation time if you are reloading the data from a previous selection.
		no_treatment = if no treatments and there is a full population of cells. 
	"""
	data_folder = os.getcwd()
	df = pd.DataFrame()
	dft = getDataTables(data_folder,not no_treatments)
	dft = cropImages(data_folder,dft,regenerate_crops = True)
	df = pd.concat([df,dft],ignore_index=True)
	print("-- DATA on "+data_folder+" LOADED --")
	return df

def getDataTables(dirSample, treatment = True):
    """
        dirSample : directory where info from CellProfiler was stored. --cp is the folder and it has to be subdivided in the folder tables and images.
        treatment : to split data by treatment. Follows from method loadData
    """
    flist = glob.glob(dirSample + "\\*--cp*")
    flistdir= [f for f in flist if os.path.isdir(f)]
    if len(flistdir)==0:
        raise Exception('--cp folder not found in :'+dirSample)
    flist2 = glob.glob(flistdir[0] + "\\*tables*")
    flistdir2= [f for f in flist2 if os.path.isdir(f)]
    if flistdir2 is None:
        raise Exception('Tables folder not found :'+dirSample)
    return getData(flistdir2[0], treatment)

def getData(mainfolder, treatment = True):
    all_data = all_data_in_one_table(mainfolder)
    all_data = all_data[_features]
    all_data = all_data.dropna()
    if treatment:
        for ind, el in enumerate(_treatments):
            all_data.loc[all_data[treatment_index]==ind,treatment_column] = el
    else:
        all_data[treatment_column]='UMO' # Unknown 
    # Remove original columns
    return all_data


def cropImages(data_sample, dft, spacing  = 50, regenerate_crops = True, color = (25,255,255)):
    # For every row, find the image
    flist = glob.glob(data_sample + "\\*--cp*")
    flistdir = [f for f in flist if os.path.isdir(f)]
    if len(flistdir) == 0:
        raise Exception('--cp folder not found in :' + data_sample)
    flist2 = glob.glob(flistdir[0] + "\\*images*")
    flistimages = [f for f in flist2 if os.path.isdir(f)]
    if flistimages is None:
        raise Exception('Images folder not found :' + data_sample)
    # Create a folder for crops
    dir_to_save = data_sample+"\\crops"
    try:
        os.makedirs(dir_to_save)
    except FileExistsError:
        if (regenerate_crops):
            shutil.rmtree(dir_to_save)
            try:
                # Try to recreate
                os.makedirs(dir_to_save, exist_ok=True)
            except OSError as e:
                # If problem is that directory still exists, wait a bit and try again
                if e.winerror == 183:
                    time.sleep(0.01)
                else:
                    raise

    dft["img_name"] = ""
    dft["img_name_raw"] = ""
    dft = dft.reset_index(drop=True)
    for row in tqdm(dft.itertuples()):
        ind  = row.Index
        imname =  dft.at[ind,'FileName_ColorImage']
        image_name = flistimages[0]+"\\"+str(imname)
        crop_name = imname[:-4] + "_" + str(ind) + ".png"
        # Load image
        if os.path.exists(image_name):
            if regenerate_crops:
                img =  cv2.imread(image_name)
                img =  cv2.normalize(img,img,30,250,cv2.NORM_MINMAX)#autoBC(img)
                x = dft.at[ind,'Location_Center_X']
                y = dft.at[ind,'Location_Center_Y']
                # Crop around center of object (90 by 90 pixels, is more than enough)
                xmin = int(np.max([0,x-spacing]))
                xmax = int(np.min([img.shape[0]-1,x+spacing]))
                ymin = int(np.max([0, y - spacing]))
                ymax = int(np.min([img.shape[1]-1, y + spacing]))
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.imwrite( dir_to_save+"\\"+ crop_name,img)
            dft.at[ind,"img_name"] = dir_to_save+"\\"+ crop_name
            dft.at[ind,"img_name_raw"] =  "./crops/"+crop_name
        else:
            print("NO-"+image_name)
            print(str(ind))
    return dft

def findTables(foldname,reg_expr):
    tables_list = []
    dirs = []
    dirs.append(foldname)
    while dirs:
        fname = dirs.pop()
        fname_t = str(fname) +"\\"+reg_expr
        flist = glob.glob(fname_t)
        if not flist:
            newdirs = ([f for f in glob.glob(fname + '\\*') if os.path.isdir(f)])
            for el in newdirs:
                dirs.append(el)
        else:
            for el in flist:
                tables_list.append(el)
    return tables_list

def all_data_in_one_table(folder_tables):
    tables_list = findTables(folder_tables,'*objects.csv*')
    tables_list_images = findTables(folder_tables,'*images.csv*')
    data_frames = []
    # Only one file containing everything
    if len(tables_list_images)==1 and  len(tables_list)==1 :
        with open(tables_list[0], 'r') as myfile:
            data = myfile.read().replace('\"', '')
        with open(tables_list[0], 'w') as myfile:
            myfile.write(data)
        with open(tables_list_images[0], 'r') as myfile:
            data = myfile.read().replace('\"', '')
        with open(tables_list_images[0], 'w') as myfile:
            myfile.write(data)
        table = pd.read_csv(tables_list[0], ',')
        images = pd.read_csv(tables_list_images[0], ',')
        # merged_df = table.join(images,on='ImageNumber',rsuffix='_other') #ImageNumber
        merged_df = pd.merge(left=table, right=images, on=('ImageNumber', 'Metadata_BaseFileName', 'Metadata_U', 'Metadata_V', 'Metadata_X', 'Metadata_Y'))
    else:
        # For each file in folder tables select objects
        #  compile the list of dataframes you want to merge
        tables_list = sorted(tables_list, key=str.lower)
        tables_list_images = sorted(tables_list_images, key=str.lower)
        for table_file,image_file in zip(tables_list,tables_list_images):
            with open(table_file, 'r') as myfile:
                data = myfile.read().replace('\"', '')
            with open(table_file, 'w') as myfile:
                myfile.write(data)
            with open(image_file, 'r') as myfile:
                data = myfile.read().replace('\"', '')
            with open(image_file, 'w') as myfile:
                myfile.write(data)
            table = pd.read_csv(table_file, ',')
            images = pd.read_csv(image_file, ',')
            # merge in big table
            ftable = table.merge(images)
            data_frames.append(ftable)
        merged_df = pd.concat(data_frames)
    return merged_df


####################################### PLOTTING ###############################################################3

from bokeh.layouts import layout
import bokeh
from bokeh.io import curdoc

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


def plotQC(df, value, title, size = 10, jitter = 0.35, factor_reduce = 0.5):
    df_i = df.dropna(subset=[value])
    df_i = df_i.reset_index(drop=True)
    key_dimensions   = [(value, title)]
    value_dimensions = [('Gene', 'Gene'),('Metadata_X', 'Position')]
    macro = hv.Table(df_i, key_dimensions, value_dimensions)
    options = dict(color_index='Position', legend_position='left', jitter=jitter, width=1000, height=600,
    scaling_method='width', scaling_factor=2, size_index=2, show_grid=True,
    tools=['hover','box_select','lasso_select'], line_color='k', cmap='Category20',size = size , nonselection_color = 'lightskyblue')
    quality_scatter = macro.to.scatter('Gene', [title]).options(**options)
    sel  = streams.Selection1D(source=quality_scatter)

    image_name = df_i.loc[0,"img_name_raw"]
    img = cv2.imread(image_name,0)
    h, w = img.shape
    w = int(factor_reduce*w)
    h = int(factor_reduce*h)
    pad = int(2.2 * w)
    def selection_callback(index):
        if not index:
            return hv.Div("")
        divtext = f'<table width={pad}  border=1 cellpadding=10 align=center valign=center>'
        for i,j in grouped(index,2):
            value_s = '{:f}'.format(df_i[value][i])
            value_s2 = '{:f}'.format(df_i[value][j])
            divtext += '<tr>'
            divtext += f'<td align=center valign=center><br> {i} Value: {value_s}</br></td>' + "\n"
            divtext += f'<td align=center valign=center><br> {j} Value: {value_s2}</br></td>' + "\n"
            divtext += '</tr><tr>'
            divtext += f'<td align=center valign=center><img src={df_i.loc[i,"img_name_raw"]} width={w} height={h}></td>'
            divtext += f'<td align=center valign=center><img src={df_i.loc[j,"img_name_raw"]} width={w} height={h}></td>'
            divtext += '</tr>'
        if len(index)%2 == 1 :
            value_s = '{:f}'.format(df_i[value][index[-1]])
            divtext += '<tr>'
            divtext += f'<td align=center valign=center><br> {index[-1]} Value: {value_s}</br></td>' + "\n"
            divtext += f'<td align=center valign=center><br> </br></td>' + "\n"
            divtext += '</tr><tr>'
            divtext += f'<td align=center valign=center><img src={df_i.loc[index[-1],"img_name_raw"]} width={w} height={h}></td>'
            divtext += f'<td align=center valign=center></td>'
            divtext += '</tr>'
        divtext += '</table>'
        return hv.Div(str(divtext))

    div = hv.DynamicMap(selection_callback, streams=[sel])
    hv.streams.PlotReset(source=quality_scatter, subscribers=[lambda reset: sel.event(index=[])])
    return  hv.Layout(quality_scatter + div).cols(1), sel

def applyQC(idf, column, vmin= None, vmax = None):
    print("QC for "+str(column))
    if vmin is None:
        vmin = -np.inf
    if vmax is None:
        vmax = np.inf
    print("Values applied :["+str(vmin)+","+str(vmax)+"].")
    print("Original number of cells:"+str(len(idf)))
    idf = idf.dropna(subset=[column])
    idf = idf[idf[column] > vmin]
    idf = idf[idf[column] < vmax]
    print("After applying control "+column+" :"+str(len(idf)))
    print("---QC "+str(column)+" done.")
    idf = idf.reset_index(drop=True)
    return idf



def getZscore(all_data,hue=None,column=None,remove_outliers=True):
    """
    :param all_data: dataframe
    :param hue: if hue is not given, each column normalizes against itself : value - mean / std
                if hue is given, then data is grouped by hue and normalized by group :  value - mean(of its group)/std(of its group)
    :param column: if column is given (hue must be given too), then the data is grouped by hue, and then grouped again by column.
                   For example we might want to group data first by study, and then by gene.
    :param remove_outliers: If True, data passing 4 times std is removed. We decided to use 4 only to remove outliers that
                            are potential numerical errors, not from the standard distribution of the data.
    :return:
    """
    z_score = all_data
    if hue is not None:
        final_df = pd.DataFrame()
        m_indexes =  list(all_data[hue].unique())
        for el in m_indexes:
            data_calc = all_data.loc[all_data[hue] == el,:].copy()
            if(column is not None):
                m_indexes2 = list(data_calc[column].unique())
                for el2 in m_indexes2:
                    data_calc2 = data_calc.loc[data_calc[column] == el2, :].copy()
                    calc_d = data_calc2.select_dtypes(include=numerics)
                    for col in calc_d:
                        if col != hue and col != column:
                            col_zscore = col + '_zscore'
                            data_calc2.loc[:, col_zscore] = (calc_d[col] - calc_d[col].mean()) / calc_d[col].std(ddof=0)
                            if remove_outliers:
                                data_calc2 = data_calc2[data_calc2[col_zscore] < 4 * data_calc2[col_zscore].std()]
                    final_df = pd.concat([final_df, data_calc2])
            else:
                calc_d = data_calc.select_dtypes(include=numerics)
                for col in calc_d:
                    if col!=hue:
                        col_zscore = col + '_zscore'
                        data_calc.loc[:,col_zscore] = (calc_d[col] - calc_d[col].mean()) / calc_d[col].std(ddof=0)
                        if remove_outliers:
                            data_calc = data_calc[data_calc[col_zscore]<4*data_calc[col_zscore].std()]
                final_df = pd.concat([final_df,data_calc])
    else:
        for col in all_data:
            col_zscore = col + '_zscore'
            z_score[col_zscore] = (z_score[col] - z_score[col].mean()) / z_score[col].std(ddof=0)
        final_df = z_score
    return final_df


def getZscoreAgainstControl(all_data, hue, control, remove_outliers = True, drop_index = True ):
    """
    :param all_data: dataframe

    :param remove_outliers: If True, data passing 4 times std is removed. We decided to use 4 only to remove outliers that
                            are potential numerical errors, not from the standard distribution of the data.
    :return:
    """
    final_df = pd.DataFrame()
    m_indexes =  list(all_data[hue].unique().astype('str') )
    query_one = ""
    for el in control:
        if el in m_indexes:
            query_one = query_one + hue + "==\'" + str(el) + "\'|"
        else:
            return
    query_one = query_one[:-1] # remove last or
    df_q = all_data.query(query_one).copy()

    if remove_outliers:
        for col in all_data.select_dtypes(include=numerics):
            df_q = df_q[(df_q[col]-df_q[col].mean()) < 4 * df_q[col].std()]

    eps = 1e-15
    for el in m_indexes:
             data_calc = all_data.query(hue+"==\'"+str(el)+"\'").copy()
             data_calc = data_calc.reset_index(drop=drop_index)
             for col in data_calc.select_dtypes(include=numerics):
                 if col!=hue and col!='index':
                         col_zscore = col + '_zscore'
                         data_calc[col_zscore] = (data_calc[col]-df_q[col].mean())/(df_q[col].std(ddof=0)+eps)
                         if remove_outliers:
                            data_calc = data_calc[data_calc[col_zscore]<4*data_calc[col_zscore].std()]
             final_df = pd.concat([final_df,data_calc])
    return final_df


def getZscoreAgainstControlPerColumn(all_data,hue,control,column):
    """
    :param all_data: dataframe

    :param remove_outliers: If True, data passing 4 times std is removed. We decided to use 4 only to remove outliers that
                            are potential numerical errors, not from the standard distribution of the data.
    :return:
    """
    final_df = pd.DataFrame()
    m_indexes =  list(all_data[hue].unique().astype('str') )
    query_one = ""
    for el in control:
        if el in m_indexes:
            query_one = query_one + hue + "==\'" + str(el) + "\'|"
        else:
            return
    query_one = query_one[:-1] # remove last or
    df_q = all_data.query(query_one).copy()

    eps = 1e-15

    if hue is not None:
        for el in m_indexes:
            data_calc = all_data.query(hue+"==\'"+str(el)+"\'").copy()
            data_calc = data_calc.reset_index(drop=True)
            if(column is not None):
                m_indexes2 = list(data_calc[column].unique())
                for el2 in m_indexes2:
                    data_calc2 = data_calc.loc[data_calc[column] == el2, :].copy()
                    data_calc2 = data_calc2.reset_index(drop=True)
                    calc_d = data_calc2.select_dtypes(include=numerics)
                    for col in calc_d:
                        if col != hue and col != column:
                            col_zscore = col + '_zscore'
                            df_q2 = df_q.query(column+"==\'"+str(el2)+"\'").copy()
                            data_calc2.loc[:, col_zscore] = (calc_d[col] -df_q2[col].mean())/(df_q2[col].std(ddof=0)+eps)
                    final_df = pd.concat([final_df, data_calc2])

    return final_df


def computeZvector(idata, hue, control, features_to_eval):
    """
    :param all_data: dataframe
    :return:
    """
    all_data = idata.copy()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    m_indexes =  list(all_data[hue].unique().astype('str') )
    query_one = ""
    for el in control:
        if el in m_indexes:
            query_one = query_one + hue + "==\'" + str(el) + "\'|"
        else:
            break
    query_one = query_one[:-1] # remove last character

    df_q = all_data.query(query_one).copy()
    eps = 1e-15

    # Compute average for each feature, per each treatment
    avg_vec = pd.DataFrame()
    for el in m_indexes:
        data_calc = all_data.query(hue+"==\'"+str(el)+"\'").copy()
        for col in data_calc.select_dtypes(include=numerics):
            if col in features_to_eval:
                avg_vec.loc[el,col] = data_calc[col].mean()

    # Compute length of vector
    all_data.loc[:,'length'] = 0
    for feature in features_to_eval:
            all_data['length'] = all_data['length'] + all_data[feature]**2

    all_data['length'] = np.sqrt(all_data['length'])

    # Compute cosine
    # Dot product of each vector per each mean v*w
    all_data.loc[:,'cosine'] = 0
    for el in m_indexes:
        for feature in features_to_eval:
            all_data.loc[all_data['Gene']==el,'cosine'] =  all_data.loc[all_data['Gene']==el,'cosine'] + all_data[all_data['Gene']==el][feature]*avg_vec.loc[el,feature]

    # Norm of avg_vec
    v_avg_norm = np.sqrt(np.sum(avg_vec**2,axis=1))

    for el in m_indexes:
        all_data.loc[all_data['Gene']==el,'cosine']= all_data.loc[all_data['Gene']==el,'cosine'] / (all_data.loc[all_data['Gene']==el,'length']*v_avg_norm[el])

    all_data['projection'] = all_data['length'] * all_data['cosine']
    return all_data


import matplotlib.cm as cm
def plotNselect_tSNE(df_feat, df, size=10, perplexity = 80, ncomp = 2):
        df_i = df.reset_index(drop=True)

        Y = TSNE(perplexity=perplexity, n_components=ncomp).fit_transform(df_feat)

        options = dict(legend_position='left', width=1000, height=600,
                       scaling_method='width', scaling_factor=2, size_index=2, show_grid=True,
                       tools=['hover', 'box_select', 'lasso_select'], line_color='k', cmap='Category20', size=size,
                       nonselection_color='lightskyblue')
        quality_scatter = hv.Scatter(Y).options(**options)

        sel = streams.Selection1D(source=quality_scatter)

        def selection_callback(index):
            divtext = ""
            for i in index:
                divtext += f'{i} <img src={df_i.loc[i,"img_name_raw"]} width=220>' + "\n"
            return hv.Div(str(divtext))

        div = hv.DynamicMap(selection_callback, streams=[sel])
        hv.streams.PlotReset(source=quality_scatter, subscribers=[lambda reset: sel.event(index=[])])
        return quality_scatter << div

def plot_tSNE(X,X_control):
    fX = pd.concat([X,X_control])
    len_f1 =len(X)
    len_f2 =len(X_control)
    labels_t= ['red']*len_f1+['blue']*len_f2
    Y = TSNE(perplexity=80,n_components=2).fit_transform(fX)
    cmap = cm.get_cmap(name='hsv')
    #Plot.scatter(Y[:, 0], Y[:, 1], s=20, alpha=0.8, c=cmap(labels / np.max(labels)))
    Plot.scatter(Y[:, 0], Y[:, 1], s=75, alpha=0.8, c=labels_t)
    classes = ['Pheno', 'Control']
    class_colours = ['r', 'b']
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    Plot.legend(recs, classes, loc=4)
    Plot.show()


def findImagesPrescan(data_folder,pattern):
    genes_dir = glob.glob(data_folder + "\\*field*")
    gene_dict = {}
    for edir in genes_dir :
        gene_name = edir[-3:]
        gene_pos  = edir[-8:-5]
        if gene_name not in gene_dict.keys():
            gene_dict[gene_name] = []
        files = glob.glob(edir+"\\*"+pattern+"*.tif")
        gene_dict[gene_name].append(dict({gene_pos:files}))
    return gene_dict

import itertools
def findImagesGene(df, data_folder,genes,genelist):
    cp_dir = glob.glob(data_folder + "\\*--cp*")[0]
    images_dir = cp_dir + "\\images"
    gene_dict = {}
    for gene in genelist:
        codelist = genes[gene]
        files = []
        for codey in codelist:
            pattern =  'Y'+str(codey).zfill(2)
            files.append(glob.glob(images_dir+"\\*"+pattern+"*.png"))
        files= list(itertools.chain(*files))
        gene_dict[gene] = files
    return gene_dict


# This method return all features given a experiment X and a gene Y
# It also renames to nicer names to plot
def getFeatures(idf_features,experiment,gene):
    features_pheno = idf_features.query('Gene==\''+gene+'\' & s_index==\''+experiment+'\'').copy()
    features_pheno = features_pheno[["Tubular","Diffuse","Fragmented","Condensed","Nuc_Solidity"]]
    return features_pheno




"""
nlcmap - a nonlinear cmap from specified levels

Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Release under MIT license.

Some hacks added 2012 noted in code (@MRR)
"""

from pylab import *
from numpy import *
from matplotlib.colors import LinearSegmentedColormap


class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        # @MRR: Need to add N for backend
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = asarray(levels, dtype='float64')
        self._x = self.levels / self.levels.max()
        self._y = linspace(0.0, 1.0, len(self.levels))

    # @MRR Need to add **kw for 'bytes'
    def __call__(self, xi, alpha=1.0, **kw):
        yi = interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)


if __name__ == "__main__":
    df2 = pd.read_csv('selected_cells.csv')
    df = loadData(no_treatments=False, regenerate_crops = False)


    print("Plot several features from cells in one 2D space using t-SNE.")
    #X = getData("D:\\ANALYSIS\\20171011_automation\\tables")
    # X = getZscore(X)
    # print(lpheno)
    # p_pheno = lpheno.index('ACTR3')
    # control = lpheno.index('COPB2')
    # features_pheno = X[X.loc[:, 'Metadata_Y'] == p_pheno ]
    # strip_plot_features(features_pheno)
    # plot_tSNE(X[X.loc[:, 'Metadata_Y'] == p_pheno],X[X.loc[:, 'Metadata_Y'] == control])




