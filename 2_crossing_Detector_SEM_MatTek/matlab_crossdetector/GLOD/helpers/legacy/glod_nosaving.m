function glod_nosaving(input_im,softParams)
    % Read preferences file and extract parameters
    % Read file
    
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

    images = cell(1,5);
    images{1} = PREPRO;
    images{2} = BWEDGE;
    images{3} = ORIENTIM;
    images{4} = RELIABILITY;
    images{5} = FSWT;
    
    gridsize = softParams.gridSize/softParams.PixelSize;
    [npeaks,error] = findBestPeaks(FSWT,ORIENTIM,softParams.K,1,gridsize,softParams.dispim);
    if(error>0)
        fprintf('ERROR')
        return
    end
    goodlines = tlines(input_im,npeaks);
    fprintf('Finding Lines by Projected Orientations completed.\n');
end



