function [finalSet]= generatePointsList(cutpoints,letter,icut,angneg)
      %% Given a letter and a subset from the cutpoints (indicated in the 3rd row) 
       xlabels='0123456789abcdefghijk';
       ylabels='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
       poslminus =  find(xlabels==letter(1));
       poslplus  =  find(ylabels==letter(2));
       
    %  we associate a list of letter names
       finalSet(length(cutpoints)).point = [0;0;0];  % i,j and group
       finalSet(length(cutpoints)).letter = '';
       
       % find all the points from the set
       cpoints = cutpoints(find(cutpoints(:,3)==icut),:);
       % find all the letters for the small set
       
       [cpoints,imset] = sortSquarePoints(cpoints,angneg);
       [cpoints,clet] = fillLetters(cpoints,letter);
       
       igpoint = find(cutpoints(:,3)==icut); 
       for i=1:4
           finalSet(igpoint(i)).point = cpoints(i,:);
           finalSet(igpoint(i)).letter = clet{i}; 
       end   
       
       % From the total number of groups
       tg = cutpoints(length(cutpoints),3); % total groups of 4
       vl = zeros(1,tg); 
       vl(icut) = 1; % The one we just fill with letters is complete
       to_check = icut;
       sdone = [];
       while(~isempty(to_check))
         nset = to_check(1); % Typical group search by not doing again the ones completed
         to_check(1) = [];
         sdone = [sdone, nset];
         igpoint = find(cutpoints(:,3)==nset); 
         gpoint = cutpoints(igpoint,:);
         [cpoints,imset] = sortSquarePoints(gpoint,angneg);
        
         % We need to find which sets are contacting
         % Group points 1 and 2
         fpoint = cpoints(1,1:2);
         spoint = cpoints(2,1:2);
         tpoint = cpoints(3,1:2);
         qpoint = cpoints(4,1:2);

         ifirst = intersect(find(cutpoints(:,1)==fpoint(1)),find(cutpoints(:,2)==fpoint(2))); % Find which points are the same in other cut group
         isecond = intersect(find(cutpoints(:,1)==spoint(1)),find(cutpoints(:,2)==spoint(2)));
         ithird = intersect(find(cutpoints(:,1)==tpoint(1)),find(cutpoints(:,2)==tpoint(2)));
         ifourth = intersect(find(cutpoints(:,1)==qpoint(1)),find(cutpoints(:,2)==qpoint(2)));

         igf = cutpoints(ifirst,3); % get the group values that have the first
         igs = cutpoints(isecond,3); % get the group value  that have the second
         igt = cutpoints(ithird,3); % get the group value  that have the third
         igq = cutpoints(ifourth,3); % get the group value  that have the fourth

         % Intersect 1 and 2
         set1 = intersect(igf,igs);
         for i=1:length(sdone)
            set1(set1==sdone(i))=[];
         end
         if(~isempty(set1))
            cpointsN = cutpoints(cutpoints(:,3)==set1,:);
            inletter = strcat(xlabels(poslminus),ylabels(poslplus-1)); 
            if(~vl(set1))% if the group is not complete, then complete it
                    [cpointsN,imset] = sortSquarePoints(cpointsN,angneg);
                    [cpointsN,clet] = fillLetters(cpointsN,inletter);
                    setpoints = find(cutpoints(:,3)==set1); 
                    for k=1:4
                        finalSet(setpoints(k)).point = cpointsN(k,:);
                        finalSet(setpoints(k)).letter = clet{k}; 
                    end
                    to_check = [to_check; set1]; % we need to check it % group has the letters, but is not checked
            end
         end
        % Intersect 2 and 3
         set1 = intersect(igs,igt);
         for i=1:length(sdone)
            set1(set1==sdone(i))=[];
         end
         if(~isempty(set1))
            cpointsN = cutpoints(cutpoints(:,3)==set1,:);
            inletter = strcat(xlabels(poslminus+1),ylabels(poslplus)); 
            if(~vl(set1))% if the group is not complete, then complete it
                    [cpointsN,imset] = sortSquarePoints(cpointsN,angneg);
                    [cpointsN,clet] = fillLetters(cpointsN,inletter);
                    setpoints = find(cutpoints(:,3)==set1); 
                    for k=1:4
                        finalSet(setpoints(k)).point = cpointsN(k,:);
                        finalSet(setpoints(k)).letter = clet{k}; 
                    end
                    to_check = [to_check; set1]; 
            end
         end
         %Intersect 3 and 4
         set1 = intersect(igt,igq);
         for i=1:length(sdone)
            set1(set1==sdone(i))=[];
         end
         if(~isempty(set1))
            cpointsN = cutpoints(cutpoints(:,3)==set1,:);
            inletter = strcat(xlabels(poslminus),ylabels(poslplus+1)); 
            if(~vl(set1))% if the group is not complete, then complete it
                    [cpointsN,imset] = sortSquarePoints(cpointsN,angneg);
                    [cpointsN,clet] = fillLetters(cpointsN,inletter);
                    setpoints = find(cutpoints(:,3)==set1); 
                    for k=1:4
                        finalSet(setpoints(k)).point = cpointsN(k,:);
                        finalSet(setpoints(k)).letter = clet{k}; 
                    end
                    to_check = [to_check; set1]; 
            end
         end
         % Intersect 4 and 1
         set1 = intersect(igq,igf);
         for i=1:length(sdone)
            set1(set1==sdone(i))=[];
         end
         if(~isempty(set1))
            cpointsN = cutpoints(cutpoints(:,3)==set1,:);
            inletter = strcat(xlabels(poslminus-1),ylabels(poslplus)); 
            if(~vl(set1))% if the group is not complete, then complete it
                    [cpointsN,imset] = sortSquarePoints(cpointsN,angneg);
                    [cpointsN,clet] = fillLetters(cpointsN,inletter);
                    setpoints = find(cutpoints(:,3)==set1); 
                    for k=1:4
                        finalSet(setpoints(k)).point = cpointsN(k,:);
                        finalSet(setpoints(k)).letter = clet{k}; 
                    end
                    to_check = [to_check; set1]; 
            end
         end

        
       end 
end
  %% Given a letter and a set of points, assign a letter value to each point 
  %  The letter is supposed to belong to the upper RIGHT corner of the
  %  square after rotating angneg (the image is then supposed to be
  %  straight) 
    function [finalSet,letters] =  fillLetters(cutpoints,letter)    
       
              
       xlabels='0123456789abcdefghijk';
       ylabels='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
       
       poslminus =  find(xlabels==letter(1));
       poslplus  =  find(ylabels==letter(2));
       % Then, based on the orientation, we decide which one is the letter,
       % 1 upper left corner, 2, upper right corner, 3 bottom left corner 4
       % bottom right corner
       j = 1;

       finalSet(j,:) =  cutpoints(2,:);
       letters{j} = letter;

       finalSet(j+1,:)  =  cutpoints(1,:);
       letters{j+1} =  strcat(xlabels(poslminus+1),ylabels(poslplus));          

       finalSet(j+2,:)   =  cutpoints(4,:);
       letters{j+2} =  strcat(xlabels(poslminus+1),ylabels(poslplus+1));
       % The most far away point is for sure the opposite      
       finalSet(j+3,:) =  cutpoints(3,:);
       letters{j+3} = strcat(xlabels(poslminus),ylabels(poslplus+1)); 
  
    end 
    

  
    
  
  %%
  function [slist,sortorder] =  sortToOrigin(plist)
    
    x0 = 0;
    y0 = 0;
    % corresponding sort-value, pick one out or make one up yourself:
    sortval = plist(:,1); % the x-value
    sortval = plist(:,2); % y-value
    sortval = (plist(:,1)-x0).^2 + (plist(:,2)-y0).^2; % distance form point (xo,y0)
    [~,sortorder] = sort(sortval);
    slist = plist(sortorder,:);
    
  end