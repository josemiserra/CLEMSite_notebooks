function [namedpoints]=assignCrownLetters(fpoints,flist,angneg)
[~,nels] = size(flist);
namedpoints = {};
listeval = [];
k = 1;
% for each point
for i=1:nels
    % find the central point in the main list
    cpoint = flist(i).point(1:2);
    ind = intersect(find(fpoints(:,1)==cpoint(1)),find(fpoints(:,2)==cpoint(2)));
    if(~any(listeval==ind))
        listeval(i) = ind;
        tmpletter = flist(i).letter; % get the letter
        namedpoints{ind} = tmpletter;
        k = k+1;
        crown_ind = fpoints(ind,3); % get the group
        crown_group = fpoints(fpoints(:,3)==crown_ind,:);    
        % take the rest of points from the group
        indcg = intersect(find(crown_group(:,1)==cpoint(1)),find(crown_group(:,2)==cpoint(2)));
        crown_group(indcg,:) = [];
        [nrows,ncols] = size(crown_group);
        [crown_group,ocg] = sortSquarePoints(crown_group,angneg);
        for j=1:nrows
            ind = intersect(find(fpoints(:,1)==crown_group(j,1)),find(fpoints(:,2)==crown_group(j,2)));
            lname = strcat(tmpletter,'_');
            lname = strcat(lname,num2str(j));
            namedpoints{ind} = lname;
        end
    end
end
end

