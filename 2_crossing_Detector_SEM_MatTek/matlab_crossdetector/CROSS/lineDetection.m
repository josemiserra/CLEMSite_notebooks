%% INPUT: folder where to store images
% 
% iim - image itself
% folder - folder where to save
% name of image  - suffix to add to all images
% Parameters for soft
 function [images,fpoints,info,error] = lineDetection(iim,softParams,gridsize,folder,imagename)

 error = 0;
if isempty(folder)
    ifolder='.'   
    folder = strcat('/softd_lm_',date);
    folder = strcat(ifolder,folder);
    mkdir(folder);
end

if isempty(imagename)
    imagename = 'image'
end

if isempty(softParams)
    softParams.sigma = 2;
    softParams.clahe = 1;
    softParams.canthresh = 0.075;
    softParams.strokeWidth = 50; 
    softParams.dispim = true;
    softParams.K = 12;
end


[PREPRO,BWEDGE,ORIENTIM,RELIABILITY,FSWT] = soft(iim,'clahe',softParams.clahe,'strokeWidth',softParams.strokeWidth,'canthresh',softParams.canthresh,'wiener',[2 2],'sigma',softParams.sigma,'dispim',softParams.dispim);

images = cell(1,5);
images{1} = PREPRO;
images{2} = BWEDGE;
images{3} = ORIENTIM;
images{4} = RELIABILITY;
images{5} = FSWT;

[iLength, iWidth] = size(PREPRO);

% Select the most prevalent angles at low resolution
npeaks = [];

Rhigh = projections(FSWT,ORIENTIM,softParams.K,1,[-45 45],10);

if(softParams.dispim)  
        THETA = (-90:1:89);
        n =length(THETA);
        [iLength, iWidth] = size(FSWT);
        iDiag = sqrt(iLength^2 + iWidth^2);
        figure, imagesc(THETA,[1:iDiag],Rhigh); colormap(pink); colorbar;
        xlabel('\theta'); ylabel('x\prime');
    end;
[npeaks,error] = getPeaks(Rhigh,gridsize);
if(error>0)
    info = [];
    fprintf('%d peaks found.Cancelling detection.\n',length(npeaks));
    if(error==3)
        fpoints=[];
        return;
    end
end    
[fpoints,angpos,angneg] = selectgridpoints(iim,npeaks,folder);
if(isempty(fpoints))
    info = []
    error = 1;
    return
end
info.angpos = angpos;
info.angneg = angneg;
info.peaks = npeaks;
fprintf('Positive angle: %d\n',angpos);
fprintf('Positive angle: %d\n',angneg);
fprintf('Finding Lines by Projected Orientations completed.\n');




fname = strcat(folder,'/prepro_');
fname = strcat(fname,imagename);
fname = strcat(fname,'.tif');
imwrite(uint8(images{1}),fname,'tif');

fname = strcat(folder,'/edge_');
fname = strcat(fname,imagename);
fname = strcat(fname,'.tif');
imwrite(images{2},fname,'tif');

fname = strcat(folder,'/ridge_');
fname = strcat(fname,imagename);
fname = strcat(fname,'.tif');
imwrite(images{3},fname,'tif');

fname = strcat(folder,'/rel_');
fname = strcat(fname,imagename);
fname = strcat(fname,'.tif');
imwrite(images{4},fname,'tif');

fname = strcat(folder,'/swt_');
fname = strcat(fname,imagename);
fname = strcat(fname,'.tif');
imwrite(images{5},fname,'tif');

fname = strcat(folder,'/R_');
fname = strcat(fname,imagename);
fname = strcat(fname,'.tif');

THETA = (-90:1:89);
n =length(THETA);
[iLength, iWidth] = size(images{1});
iDiag = sqrt(iLength^2 + iWidth^2);
figure('Visible','off');
imagesc(THETA,[1:iDiag],Rhigh); colormap(hot); colorbar;
xlabel('\theta'); ylabel('x\prime');
saveas(gcf,fname,'tif');
end
 

 
 