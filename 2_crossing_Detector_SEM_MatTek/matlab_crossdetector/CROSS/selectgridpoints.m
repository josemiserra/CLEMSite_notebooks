function [fpoints,angpos,angneg] = selectGridPoints(iimg,ipeaks,folder)
%% This MATLAB function takes all the lines from an image
% and calculate the intersection between lines
% and the crossing point.
% The first step is removing wrong lines based on possible wrong peaks

[iLength, iWidth] = size(iimg);
angpos = ipeaks(find(ipeaks(:,2)>0),2); % first pos
angpos = angpos(1);
angneg = ipeaks(find(ipeaks(:,2)<1),2); % last neg
angneg = angneg(1);

fname = strcat(folder,'\lines');
goodlines = tlines(iimg,ipeaks,fname);

%% Peak recalibration 
% The idea is that we know the squares distance and we can refit the 
% current lines to the distances given by the manufacturer
% However, in practice is a disaster because the distances are varying and
% they oscillate between 20-40 um for lines and 550 and 570 um for squares.
% The reason could be that the manufacturer measures are innaccurate or 
% that the way of the reflected light incides with the borders can change
% the width of the borders from the top to the bottom of the images,
% causing a non-straight line.
% sqShort = 0;
% sqLong = 0;
% if(isempty(scale))
%     [sqShort sqLong] = squaresDistance(ipeaks); 
% else
%      sqShort =   round(scale*40); % given in um/px and 40 um is the known distance for the small space
%      sqLong  =   round(scale*560);
% end  
% peaks = recalibrate_peaks(ipeaks,sqShort,sqLong);

% In order to avoid the following strategy has been designed>
%  1) Find all intersection points between lines
%  2) Cluster them by groups of four
%  3) Find the intersection and redetect the lines


% find the least x coordinate of point 1
% rotate given the positive angle of this point
 
[fpoints] = calibrateIntersections(iimg,goodlines,ipeaks,folder);
if(isempty(fpoints))
    return 
end
k = 1;
for i=5:5:length(fpoints)
    cpoints(k,:) = fpoints(i,:);
    plot(cpoints(k,1),cpoints(k,2),'.','LineWidth',2,'Color','blue');
    k = k+1;
end


 fname = strcat(folder,'/sketch.tif');
 saveas(gcf,fname,'tif');

 fpoints = cpoints

end


