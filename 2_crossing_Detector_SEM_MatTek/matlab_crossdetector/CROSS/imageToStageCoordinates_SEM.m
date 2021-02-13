function [tPoints]=imageToStageCoordinates_SEM(fpoints,tpoint,iimg,pixelSize)    
    [iLength, iWidth] = size(iimg);
    icx = round(iWidth/2);
    icy = round(iLength/2);
    
    [nrows,ncols] = size(fpoints);
    tPoints = [];
    % Considered orientation of microscope
    % Direction in ++ and --
    fpoints(:,1)= fpoints(:,1)-icx;
    fpoints(:,2)= fpoints(:,2)-icy;
   
    for i=1:nrows
       tPoints(i,1) =  fpoints(i,1)*pixelSize + tpoint(1,1);
       tPoints(i,2) =  -fpoints(i,2)*pixelSize + tpoint(1,2);
       tPoints(i,3:ncols) = fpoints(i,3:ncols);
    end