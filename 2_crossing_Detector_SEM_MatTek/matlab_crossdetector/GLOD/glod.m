function [images,fpoints,info,error]= glod(input_im,softParams,folder,imagename,p_angles)
    % Read preferences file and extract parameters
    % Read file
    images = cell(1,5);
    fpoints = [];
    info = [];
    error = 0;
    
if isempty(softParams)
    softParams.sigma = 1.5;
    softParams.clahe = 1;
    softParams.canthresh = 0.075;
    softParams.strokeWidth = 25; 
    softParams.dispim = true;
    softParams.wiener = [2 2];
    softParams.K = 12;
    softParams.gridSize = [40 560]; 
    softParams.PixelSize = 1.5137; % in um/pixel
end   
   [PREPRO,BWEDGE,ORIENTIM,RELIABILITY,FSWT] = soft(input_im,'clahe',softParams.clahe,'strokeWidth',softParams.strokeWidth,'canthresh',softParams.canthresh,'wiener',softParams.wiener,'sigma',softParams.sigma,'dispim',softParams.dispim);

    
    images{1} = PREPRO;
    images{2} = BWEDGE;
    images{3} = ORIENTIM;
    images{4} = RELIABILITY;
    images{5} = FSWT;
    
    gridsize = softParams.gridSize/softParams.PixelSize;
    [npeaks,R,error] = findBestPeaks(FSWT,ORIENTIM,softParams.K,1,gridsize,softParams.dispim,p_angles);
    if(error>0)
        fprintf('ERROR')
        return
    end
    images{6} = R;
    
   
    goodlines = tlines(input_im,npeaks);
      
    show(input_im); 
    hold on;
    for k=1:2:length(goodlines)
        p1 = goodlines(k,:);
        p2 = goodlines(k+1,:);
        line( [p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'Color','red');
    end;
    fname = strcat(folder,'/lines_sketch');
    fname = strcat(fname,'.tif');   
    set(gca,'position',[0 0 1 1]);  
    saveas(gcf,fname,'tif');
    fprintf('Finding Lines by Projected Orientations completed.\n');
     
    
    
    [mpoints] = calibrateIntersections(input_im,goodlines,npeaks,folder);
    [fpoints,all_cutpoints,centrePointIndex] = selectgridpoints(input_im,mpoints,gridsize);
    angleposind = find(npeaks(:,2)>=0);
    anglenegind = find(npeaks(:,2)<0);
    info.angpos = npeaks(angleposind(1),2);
    info.angneg = npeaks(anglenegind(1),2);
    info.cutpoints = all_cutpoints;
    info.centrePointIndex = centrePointIndex;
    info.peaks = npeaks;
    fprintf('Positive angle: %d\n',info.angpos);
    fprintf('Negative angle: %d\n',info.angneg);
    
    
    
    %% Saving files   
    fprintf('Saving information.\n');
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
        imagesc(THETA,[1:iDiag],images{6}); colormap(hot); colorbar;
        xlabel('\theta'); ylabel('x\prime');
        saveas(gcf,fname,'tif');
 
        filename = strcat(imagename,'_peaks.csv');
        filename = strcat(folder,filename);
        csvwrite(filename,info.peaks);

        filename = strcat(imagename,'_fpoints.csv');
        filename = strcat(folder,filename);
        csvwrite(filename,fpoints);
         
        filename = strcat(imagename,'_cutpoints.csv');
        filename = strcat(folder,filename);
        csvwrite(filename,info.cutpoints);
    
    
    
end



