
MATLAB implementation of line detector
--------------------------------------
This software was developed originally for the paper __________

It was posteriorly replaced by a python implementation.
If you are interest to use any of the functions, contact to:

Corresponding author : Jose Miguel Serra Lleti. serrajosemi@gmail.com

The package is split in two main programs: CROSS and GLOD

GLOD, Grid Line Orientation based Detector :

Given an image, a file describing the grid, image metadata parameters (x,y stage coordinates) it performs line detection and crops the center squares out.

The two main functions to call are runLD_SEM and runLD_LM

- runLD_SEM(input_image,output_folder,parameters_filename,image_metadata,scripts_folder,true_letter)
- runLD_LM(input_image,output_folder,parameters_filename,image_metadata,scripts_folder,true_letter)

Example GLOD, executed by command line:
LM
-----
It is convenient add the paths were scripts are:
addpath(genpath('Z:\CLEMSITE\matlab_msite\GLD\')); 

runLD_LM('Z:\AUTOCLEM\07062018_nospots_automation\renamed\hr\field--X00--Y05_0003\grid_0003_3--LM--RL - Reflected Light--10x--z1.tif','Z:\AUTOCLEM\07062018_nospots_automation\renamed\hr\field--X00--Y05_0003','Z:\CLEMSITE\msite_27.05.2018-16-37-48\CLEMSite\common\user.pref','Z:/AUTOCLEM/07062018_nospots_automation/renamed/hr\field--X00--Y05_0003\field--X00--Y05_0003_info.txt','Z:\CLEMSITE\msite_27.05.2018-16-37-48\matlab_msite\GLD\'); catch; end; quit" xe"" -minimize -noFigureWindows -nosplash



SEM
----
matlab -wait -r "try; addpath(genpath('C:\Users\Documents\msite\matlab_msite\GLD\')); 

runLD_SEM('D:\TESTS\nav__201805211554595231.tif','D:\TESTS\p0_20180521_155605','C:\Users\Documents\msite\CLEMSite\common\user.pref','D:/TESTS\p0_20180521_155605\p0_20180521_155605_info.txt','C:\Users\Documents\msite\matlab_msite\GLD\'); catch; end; quit" xe"" -minimize -noFigureWindows -nosplash


## CROSS 

CROSS is the cross detector. It detects 4 peaks and the center point in the image. Here is the call by command line.

- runLD_CROSS(input_im,output_folder,tag,image_metadata,igridsize)

matlab -wait -r 

"try; 

addpath(genpath('C:\Users\Documents\msite\matlab_msite\CROSS\')); 

runLD_CROSS('D:\GOLGI\19_October_2017\Project19\3N_field--X03--Y20_0038___0007\ref_0_4O_201711082316008814.tif',
'D:\GOLGI\19_October_2017\Project19\3N_field--X03--Y20_0038___0007',
'ref_0_4O','D:\GOLGI\19_October_2017\Project19\3N_field--X03--Y20_0038___0007\4O_info.txt','40');

 catch; end; quit" xe"" -minimize -noFigureWindows -nosplash


Look at the runtest.m to see an example of inputs.






