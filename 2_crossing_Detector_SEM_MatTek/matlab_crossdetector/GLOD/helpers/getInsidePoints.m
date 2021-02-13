function [inPoints,indices]=getInsidePoints(fpoints, iimg)
   
 [iLength, iWidth] = size(iimg);
 [nrows,ncols] = size(fpoints);
 inPoints = [];
 k = 1;
 for i=1:nrows
    if((fpoints(i,1)<iWidth)&&(fpoints(i,1)>0)&&(fpoints(i,2)<iLength)&&(fpoints(i,2)>0))
        inPoints(k,:) = fpoints(i,:);
        indices(k) = i;
        k=k+1;
    end
 end
    