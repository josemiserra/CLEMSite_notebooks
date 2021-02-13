function [fpoints,all_cutpoints,cutpoint] = selectGridPoints(input_im,fpoints,gridsize)
%% This MATLAB function takes all the points
% and evaluates all possible combinations between them
% if they form a square similar to the the dimensions of the grid
% they are selected as a group of four.
[iLength,iWidth] = size(input_im);
k = 1;
for i=5:5:length(fpoints)
    cpoints(k,:) = fpoints(i,:);
    k = k+1;
end

%% Select points and crop letters
% [cp]=crop_letters()

cpoints = sortrows(cpoints,1); % order by x coordinate
% take distance of closest points (minimum distance must be a square
distsq = max(gridsize);

combos = nchoosek(1:length(cpoints),4);
% Select from the inner points the one closest to the center.
% odist = inf;
cxp = iWidth/2;
cyp = iLength/2;

ccombos = []; 
m = 1;
tolerance = 0.2*distsq;
[rows cols] = size(combos);
for k=1:rows
           set1 = cpoints(combos(k,:),1:2);
           cdist = pdist(set1);
           j = 0;
           for i=1:length(cdist)
                if((distsq-tolerance)<cdist(i) && cdist(i)<(distsq+tolerance)) 
                   j = j+1;
                end
           end
           if(j==4)
               ccombos(m,:) = combos(k,:);
               m = m+1;
           end;
end;

% Find the reference square
 all_cutpoints = [];
 cutpoint = -1;
 [rows cols] = size(ccombos);
 for i=1:rows
        if(is_insquare(cpoints(ccombos(i,:),:),[cxp;cyp])) % is the center square
            cutpoint = i;
        end
        cpoints(ccombos(i,:),3)=i; % Assign group
        all_cutpoints = [ all_cutpoints; cpoints(ccombos(i,:),:) ];
 end
 

end

 %% 

function is_in=is_insquare(points,cp)
   is_in = 0;
   points = sortrows(points,1);
   if(points(1,1)<cp(1))
       if(points(4,1)>cp(1))
           points = sortrows(points,2);
              if(points(1,2)<cp(2))
                if(points(4,2)>cp(2))
                       is_in = 1;
                end
              end
       end
   end
end
