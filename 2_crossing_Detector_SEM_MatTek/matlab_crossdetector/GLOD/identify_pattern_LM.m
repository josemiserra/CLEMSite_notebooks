function [flist,center_pattern]= identify_pattern_LM(iimg,all_cutpoints,distsq,folder,orientation,indpattern,epattern,codebookdir,true_pattern)
%%
% patternIndex - from all the datasets coming from cutpoints, which one is
% the associated with the known pattern.
% epattern - estimated pattern
%
[iLength, iWidth] = size(iimg);
[rows cols] = size(all_cutpoints);
    
%% 
k = 1;
myimg=imrotate(iimg,orientation,'crop');
show(myimg);
for j=1:4:rows
    % For each pattern, crop the image
    % Rotate inner point and move to center of coordinates
    minxp = 1;
    minyp = 1;
    distmin = Inf;
    mset = all_cutpoints(j,3);
    hold on;   
    cancelcut = 0;
    for i = 1:4        
        xp = all_cutpoints(j+i-1,1);
        yp = all_cutpoints(j+i-1,2);
        
        if((xp>iWidth+30)||(xp<-30)||(yp>iLength+30)||(yp<-30)) % We give some margin +/-30 pix to accept image
            cancelcut = 1;
        end
        minpoint = [xp-iWidth/2 yp-iLength/2];
        irotmat =[ cosd(orientation) sind(orientation); -sind(orientation) cosd(orientation) ];
        minpoint = irotmat*minpoint';
        minpoint = ceil([iWidth/2+minpoint(1)+1 iLength/2+minpoint(2)+1]);
        plot(minpoint(1),minpoint(2),'.','LineWidth',2,'Color','blue');
        distp = round(sqrt(minpoint(1)^2 + minpoint(2)^2));
        if(distp<distmin) 
            minxp =minpoint(1); minyp = minpoint(2); 
            distmin = distp;
        end   
    end
    if(~cancelcut) %if we have the image from the cutpoint
        cimg = imcrop(myimg,[minxp minyp distsq distsq]);
        hold off
        
        images(k).image = cimg; % save the original for later
        images(k).set = mset ;
        cimg=imrotate(cimg,orientation,'crop');
        [nx,ny,dim] = size(cimg);
        if(dim>1)
          cimg = rgb2gray(cimg(:,:,1:3));
        end    
        if(nx~=256 || ny~=256)
           cimg = imresize(cimg,[256 256]);
        end;  
        k=k+1;
    end
end
        

%%
icpattern = indpattern;
if(~true_pattern)
     % if we only have 1 letter, we keep the original estimation and we
     % don´t risk it to the image analysis. In cases with noisy images
     % the probability of being wrong is very high with a high confidence.
     % In that cases, we want to keep our original estimation...
     list_neighs = calculateNeighs(epattern,1);
     [npattern,p_k] = classifyPattern(list_neighs,images(icpattern).image,codebookdir); % predict the letter name naively
     j = 1;
     while(~strcmp(epattern,npattern{j})&& j<=length(npattern))
         j=j+1;
     end
     fprintf('Probability of being the estimated pattern %s : %.4f \n ',epattern,p_k(j));
     old_pk = p_k(j);
     if(strcmp(epattern,npattern{1}))
         fprintf('Keeping original estimation');
     else
        if(length(images)==1)
           if(p_k(j)<0.05) % ridiculously low, less than 0.98
             old_pk = p_k(1);
             epattern = npattern{1};
           end    
        else    
        % My pattern is distinct than the estimated
        % and we have neighbors to check
        % We evaluate the probability of the estimated pattern
         old_pk = p_k(1);
         epattern = npattern{1};
         for k=1:length(images)
             [npattern,p_k] = classifyPattern(list_neighs,images(k).image,codebookdir); % predict the letter name naively
             if(p_k(1)>old_pk)
                    icpattern = images(k).set;
                    epattern = npattern{1};
                    old_pk = p_k(1);
             end
         end
        end
         fprintf('Evaluating patterns with best probabilities. Selected final pattern: %s : %.4f \n ',epattern,old_pk);
    end
end
    points_list = fillPatternNames_LM(all_cutpoints,epattern,icpattern,orientation);
    center_pattern = epattern; % just in case, initialization
    for k=1:length(images)
        points_set = points_list(find(all_cutpoints(:,3)==images(k).set));
        lettername =  points_set(1).letter;
        if(indpattern == images(k).set)
            center_pattern = lettername;
        end
        fname = strcat('\',lettername);
        fname = strcat(fname,'_');
        fname = strcat(fname,int2str(k));
        fname = strcat(folder,fname);
        fname = strcat(fname,'.tif');
        imwrite(images(k).image,fname);
        
        show(images(k).image);
    end
    
    flist = removeRepeated(points_list);

end
function [nlist] = removeRepeated(plist)  
    letters = unique(cat(1,plist(:).letter),'rows');
    k = 1;
    for i=1:length(plist)
       if(mod(i,4)==1)
          nlist(k).point  =  plist(i).point(1:2);
          nlist(k).letter =  plist(i).letter;
          nlist(k).has_image = 1;
          k = k+1;
          for j=1:length(letters)
                if(strcmp(plist(i).letter,letters(j,:)))
                    lout = j;
                    break;
                end
          end
          letters(lout,:) = [];
       end
    end
    for i=1:length(plist)
        lout = -1;
        for j=1:length(letters)
            if(strcmp(plist(i).letter,letters(j,:)))
                nlist(k).point  =  plist(i).point(1:2);
                nlist(k).letter =  plist(i).letter;
                nlist(k).has_image = 0;
                k = k+1;
                lout = j;
                break;
            end
        end
        if(lout>0)
            letters(lout)=j;
        end
    end
 end

function [copylabels] = calculateNeighs(cletter,nneig)
    %% Generate closest set of points, given a letter, which letters are around mine
    xlabels='0123456789abcdefghijk+';
    ylabels='ABCDEFGHIJKLMNOPQRSTUVWXYZ+';

    poslminus =  find(xlabels==cletter(1));
    poslplus  =  find(ylabels==cletter(2));
    k = 1;
    for i=poslminus-nneig:poslminus+nneig
        if(i<length(xlabels)&& i>0)
            minusl = xlabels(i);
            for j=poslplus-nneig:poslplus+nneig
                if(j<length(ylabels)&& j>0)
                    plusl = ylabels(j);
                    copylabels{k} = strcat(minusl,plusl);
                    k=k+1;
                end
            end
        end
    end
end